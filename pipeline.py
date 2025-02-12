"""
pipeline.py

This file contains the integrated training and feature extraction pipeline.
It combines data processing, model training (with additional accuracy metrics), model conversion,
and image embedding extraction into one function: run_pipeline().

Key changes compared to the initial version:
  - Uses the teacher’s FineTunedResNet (imported from a_resnet_transfer_trainer.py) as the model.
  - The training loop computes and logs additional metrics (accuracy).
  - After training, the trained model is converted for feature extraction by replacing its classification head (new_layers)
    with an Identity mapping so that the model outputs raw feature embeddings.
  - The CSV is read using the "labels" column (because ProductLabeler outputs columns "Image" and "labels").
  - The split DataFrames are written to temporary CSV files so that ImageDataset (which expects a file path)
    works correctly.
  - Final outputs are saved to:
       • Feature extraction model: data/final_model/image_model.pt
       • Image embeddings: data/output/image_embeddings.json
"""

import os
import pandas as pd
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.tensorboard import SummaryWriter

# Import the teacher's model and our dataset class.
from a_resnet_transfer_trainer import FineTunedResNet
from image_dataset_pytorch import ImageDataset

def create_model_dir(base_dir='data/model_evaluation'):
    """
    Create a timestamped directory (with a subfolder for weights) to store model outputs.
    """
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_dir = os.path.join(base_dir, f'model_{timestamp}')
    weights_dir = os.path.join(model_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    return model_dir, weights_dir

def run_pipeline():
    """
    Executes the full training and feature extraction pipeline.
    
    Steps:
      a) Data Processing & Setup:
         - Reads the training CSV (with columns "Image" and "labels").
         - Creates encoder/decoder mappings.
         - Splits the dataset (using stratification on "labels") into training, validation, and test sets.
         - Writes each split to temporary CSV files.
      b) Model Training:
         - Instantiates the teacher’s FineTunedResNet model.
         - Trains the model using a standard training loop while computing and logging additional metrics (accuracy).
      c) Model Conversion:
         - Replaces the classification head (new_layers) with an Identity mapping so that the model outputs raw feature embeddings.
         - Saves the converted model to 'data/final_model/image_model.pt'.
      d) Embedding Extraction:
         - Uses the feature extraction model to compute embeddings for images from the training split.
         - Saves the embeddings as a JSON file in 'data/output/image_embeddings.json'.
    """

# a) Data Processing & Setup
    csv_path = 'data/training_data.csv'
    dataframe = pd.read_csv(csv_path, dtype={'labels': int})
    
    # Use the "labels" column (ProductLabeler creates "Image" and "labels")
    unique_labels = dataframe['labels'].unique()
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    label_decoder = {idx: label for label, idx in label_encoder.items()}

    with open('image_decoder.pkl', 'wb') as f:
        pickle.dump(label_decoder, f)
    print("Decoder mapping saved to image_decoder.pkl")

    # Split the DataFrame using stratification on "labels"
    df_training, df_temp = train_test_split(dataframe, test_size=0.4, stratify=dataframe['labels'])
    df_validation, df_test = train_test_split(df_temp, test_size=0.5, stratify=df_temp['labels'])

    # Create a folder to store temporary CSV files
    temp_csv_dir = 'data/temp_csv'
    os.makedirs(temp_csv_dir, exist_ok=True)

    # Write temporary CSV files (ImageDataset expects a CSV file path)
    temp_train_csv = os.path.join(temp_csv_dir, 'temp_train.csv')
    temp_val_csv = os.path.join(temp_csv_dir, 'temp_val.csv')
    temp_test_csv = os.path.join(temp_csv_dir, 'temp_test.csv')
    df_training.to_csv(temp_train_csv, index=False)
    df_validation.to_csv(temp_val_csv, index=False)
    df_test.to_csv(temp_test_csv, index=False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Create datasets from the temporary CSV files.
    train_dataset = ImageDataset(temp_train_csv, 'cleaned_images/')
    validation_dataset = ImageDataset(temp_val_csv, 'cleaned_images/')
    test_dataset = ImageDataset(temp_test_csv, 'cleaned_images/')

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # b) Model Training
    model_training = FineTunedResNet(len(label_encoder))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_training.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model_training.parameters()), lr=0.001)

    writer = SummaryWriter('resource/tensorboard')
    model_dir, weights_dir = create_model_dir()

    def train(model, epochs=1):
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            total_loss = 0.0
            train_correct = 0
            train_total = 0

            # Unpack only two values: (images, labels)
            for i, (images, labels, img_id) in enumerate(train_dataloader):
                images = images.to(device)
                labels = labels.to(device)
                #img_id = img_id.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                # Calculate training accuracy.
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                if i % 10 == 9:
                    writer.add_scalar('training loss', running_loss / 10, epoch * len(train_dataloader) + i)
                    running_loss = 0.0

            avg_train_loss = total_loss / len(train_dataloader)
            train_accuracy = train_correct / train_total
            writer.add_scalar('avg training loss', avg_train_loss, epoch)
            writer.add_scalar('train accuracy', train_accuracy, epoch)

            # Validation phase: compute loss and accuracy.
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for images, labels, img_id in validation_dataloader:
                    images = images.to(device)
                    labels = labels.to(device)
                    #img_id = img_id.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            avg_val_loss = val_loss / len(validation_dataloader)
            val_accuracy = val_correct / val_total
            writer.add_scalar('validation loss', avg_val_loss, epoch)
            writer.add_scalar('validation accuracy', val_accuracy, epoch)

            print(f'Epoch [{epoch + 1}/{epochs}], Avg Train Loss: {avg_train_loss:.4f}, '
                  f'Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
                  f'Validation Accuracy: {val_accuracy:.4f}')

            weights_path = os.path.join(weights_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), weights_path)

            metrics_path = os.path.join(model_dir, 'metrics.txt')
            with open(metrics_path, 'a') as f:
                f.write(f'Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}, '
                        f'Train Accuracy: {train_accuracy:.4f}, Validation Loss: {avg_val_loss:.4f}, '
                        f'Validation Accuracy: {val_accuracy:.4f}\n')

        writer.flush()

    train(model_training, epochs=10) # TRAIN for 10 epochs
    writer.close()

    # c) Model Conversion for Feature Extraction
    # The teacher’s FineTunedResNet builds a combined model: self.combined_model = nn.Sequential(self.model, self.new_layers)
    
    # d) Embedding Extraction
    # Set device for PyTorch computations (GPU if available, else CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters (adjust as needed)
    num_classes = 13  # Number of unique labels/classes in the dataset
    saved_weights = 'data/final_model/image_model.pt'  # Model weights file
    training_csv = 'data/training_data.csv'  # CSV containing "Image" and "labels" columns
    image_dir = 'cleaned_images/'  # Directory containing image files

    # Step 1: Instantiate and Load Pretrained Model
    """
    Load the FineTunedResNet model trained for image classification.
    Then, convert the model to extract feature embeddings instead of classification outputs.
    """
    model_training = FineTunedResNet(num_classes)

    # Load the saved model weights
    model_training.load_state_dict(torch.load(saved_weights, map_location=device))
    model_training.to(device)

    # Convert the classification model into a feature extraction model
    # We remove the classification head to retain only feature extraction layers
    model_extractor = nn.Sequential(*list(model_training.combined_model.children())[:-1])
    model_extractor.to(device)
    model_extractor.eval()  # Set the model to evaluation mode

    # Step 2: Load Dataset
    """
    Load the dataset for embedding extraction.
    The ImageDataset class is used to load images based on a CSV file.
    """
    dataset = ImageDataset(training_csv, image_dir)

    # Create a DataLoader for efficient batch processing
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Step 3: Compute Embeddings
    """
    Iterate through images, pass them through the feature extractor, and store embeddings.
    Each embedding corresponds to a high-dimensional numerical representation of an image.
    """
    image_embeddings = {}

    with torch.no_grad():  # Disable gradient computation for inference
        for idx, (image, label, img_name) in enumerate(dataloader):
            image = image.to(device)  # Move image tensor to the correct device

            # Extract feature embedding using the modified model
            embedding = model_extractor(image)
            embedding = embedding.flatten().detach().cpu().numpy()  # Convert tensor to NumPy array

            # Store the embedding using the image filename as the key
            image_embeddings[str(img_name)] = embedding.tolist()

    # Step 5: Save Embeddings to JSON File
    """
    Save the computed image embeddings as a JSON file for future use.
    """
    output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    embeddings_path = os.path.join(output_dir, 'image_embeddings.json')
    with open(embeddings_path, 'w') as f:
        json.dump(image_embeddings, f)

    print(f"Image embeddings successfully saved to {embeddings_path}")

if __name__ == "__main__":
    run_pipeline()
