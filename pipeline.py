"""
pipeline.py

This file contains the integrated training and feature extraction pipeline.
It combines data processing, model training, conversion to a feature extraction model,
and image embedding extraction into one function: run_pipeline().

Key changes compared to the initial version:
  - The fc layer of ResNet-50 is replaced directly with a sequential block to avoid applying adaptive pooling twice.
  - Additional metrics (training and validation accuracy) are computed and logged to TensorBoard as well as saved in a metrics file.
  - After training, the trained model is converted for feature extraction by re-instantiating the same architecture, loading the trained weights,
    and then replacing the fc layer with an Identity mapping so that the network outputs raw feature embeddings.
  - The CSV is read using the "labels" column since ProductLabeler outputs columns "Image" and "labels".
  - Final outputs are saved to:
       • Feature extraction model: data/final_model/image_model.pt
       • Image embeddings: data/output/image_embeddings.json
"""

import os
import pandas as pd
import time
from PIL import Image
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import pickle
from torchvision.models import ResNet50_Weights
from torch.utils.tensorboard import SummaryWriter
from a_resnet_transfer_trainer import FineTunedResNet

# -------------------------------
# Dataset Definition
# -------------------------------
class CustomDataset(Dataset):
    """
    Custom dataset to load images and labels from a DataFrame.
    """
    def __init__(self, dataframe, root_dir, transform=None, file_extension='.jpg'):
        self.dataframe = dataframe
        self.root_dir = root_dir
        self.transform = transform
        self.file_extension = file_extension

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # The CSV has an "Image" column that serves as the image identifier.
        img_id = str(self.dataframe.iloc[idx, 0])
        img_name = os.path.join(self.root_dir, img_id + self.file_extension)
        print(f"Loading image: {img_name}")
        try:
            image = Image.open(img_name).convert('RGB')
        except FileNotFoundError as e:
            print(f"File not found: {img_name}")
            raise e
        # The label is stored in the "labels" column.
        label = self.dataframe.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        return image, label, img_id

def create_model_dir(base_dir='data/model_evaluation'):
    """
    Create a timestamped directory (with a subfolder for weights) to store model outputs.
    """
    timestamp = time.strftime('%Y%m%d-%H%M%S')
    model_dir = os.path.join(base_dir, f'model_{timestamp}')
    weights_dir = os.path.join(model_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    return model_dir, weights_dir

# -------------------------------
# Integrated Pipeline Function
# -------------------------------
def run_pipeline():
    """
    Executes the full training and feature extraction pipeline.
    
    Steps:
      a) Data Processing & Setup: Load the training CSV, create encoder/decoder mappings, and split the dataset.
      b) Model Training: Train a ResNet-50 with a revised fc layer. Additional metrics (accuracy) are computed and logged.
      c) Model Conversion: Load the trained weights and replace the fc layer with Identity for feature extraction.
      d) Embedding Extraction: Extract and save image embeddings to a JSON file.
    
    Final outputs:
      - Feature extraction model is saved to 'data/final_model/image_model.pt'
      - Image embeddings are saved to 'data/output/image_embeddings.json'
    """
    # a) Data Processing & Setup
    csv_path = 'data/training_data.csv'
    # Read the CSV; note that the CSV contains columns "Image" and "labels"
    dataframe = pd.read_csv(csv_path, dtype={'labels': int})
    
    # --- Use the "labels" column (not "category_label")
    unique_labels = dataframe['labels'].unique()
    # --------------------------------------------------------------
    
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    label_decoder = {idx: label for label, idx in label_encoder.items()}

    with open('image_decoder.pkl', 'wb') as f:
        pickle.dump(label_decoder, f)
    print("Decoder mapping saved to image_decoder.pkl")

    # Split the dataset using stratification on the labels.
    df_training, df_temp = train_test_split(
        dataframe, test_size=0.4, stratify=dataframe['labels'])
    df_validation, df_test = train_test_split(
        df_temp, test_size=0.5, stratify=df_temp['labels']
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    train_dataset = CustomDataset(
        dataframe=df_training,
        root_dir='cleaned_images/',
        transform=transform,
        file_extension='.jpg'
    )
    validation_dataset = CustomDataset(
        dataframe=df_validation,
        root_dir='cleaned_images/',
        transform=transform,
        file_extension='.jpg'
    )
    test_dataset = CustomDataset(
        dataframe=df_test,
        root_dir='cleaned_images/',
        transform=transform,
        file_extension='.jpg'
    )

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # b) Model Setup & Training
    
    # Load the pre-trained ResNet-50 model.
    model_training = FineTunedResnet()

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

            for i, (images, labels, image_ids) in enumerate(train_dataloader):
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                total_loss += loss.item()

                # Calculate training accuracy for this batch.
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)

                if i % 10 == 9:
                    writer.add_scalar('training loss',
                                      running_loss / 10,
                                      epoch * len(train_dataloader) + i)
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
                for images, labels, _ in validation_dataloader:
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

            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                  f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            weights_path = os.path.join(weights_dir, f'epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), weights_path)

            metrics_path = os.path.join(model_dir, 'metrics.txt')
            with open(metrics_path, 'a') as f:
                f.write(f'Epoch {epoch + 1}, Avg Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, '
                        f'Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}\n')

        writer.flush()

    train(model, 5)
    writer.close()

'''
    # c) Convert and Save Feature Extraction Model
    final_model_dir = 'data/final_model'
    os.makedirs(final_model_dir, exist_ok=True)
    # IMPORTANT: Instantiate the model with the same fc architecture as used during training.
    model = FineTunedResnet()
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the last fc layer
    saved_weights_path = os.path.join(weights_dir, 'epoch_5.pth') # LOOK AT THE FILEPATH
    model.load_state_dict(torch.load(saved_weights_path))

    trained_model_path = os.path.join(weights_dir, 'epoch_10.pth')
    feature_extractor_model.load_state_dict(torch.load(trained_model_path))

    # d) Extract & Save Image Embeddings
    image_embeddings = {}
    with torch.no_grad():
        for images, _, image_ids in DataLoader(train_dataset, batch_size=32, shuffle=False):
            embeddings = feature_extractor_model(images)
            for img_id, embedding in zip(image_ids, embeddings):
                image_embeddings[img_id] = embedding.tolist()

    output_dir = 'data/output'
    os.makedirs(output_dir, exist_ok=True)
    embeddings_path = os.path.join(output_dir, 'image_embeddings.json')
    with open(embeddings_path, 'w') as f:
        json.dump(image_embeddings, f)
    print(f'Image embeddings have been successfully saved to {embeddings_path}')
'''

# Allow running the pipeline directly.
if __name__ == "__main__":
    run_pipeline()