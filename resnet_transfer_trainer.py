"""
resnet_transfer_trainer.py

This module defines the ResNetTransferLearner class, which leverages transfer learning using
a pre-trained ResNet-50 model. The class encapsulates the full training pipeline:
  1. Process raw product and image data (using ProductLabeler) to generate a training CSV.
  2. Create encoder/decoder mappings and save the decoder as "image_decoder.pkl".
  3. Set up DataLoaders by splitting the dataset into training and validation sets.
  4. Configure the pre-trained ResNet-50 model by replacing its final fully connected layer.
  5. Train the model using a standard training loop with validation.
     At the end of every epoch, the model weights and metrics are saved.
"""

import os
import pickle
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset

class ResNetTransferLearner:
    """
    A class for performing transfer learning with a pre-trained ResNet-50 model.

    This class processes the data, sets up DataLoaders, configures the model, and trains it.
    The decoder (mapping numeric labels back to category names) is saved as a pickle file for future use.
    Model weights and metrics are saved at the end of every epoch.
    """
    
    def __init__(self, 
                 products_csv="data/Cleaned_Products.csv", 
                 images_csv="data/Images.csv", 
                 training_csv="data/training_data.csv", 
                 image_dir="cleaned_images/",
                 decoder_path="image_decoder.pkl",
                 batch_size=32,
                 num_epochs=5,
                 learning_rate=0.001,
                 momentum=0.9,
                 num_workers=4):
        """
        Initialize the ResNetTransferLearner with file paths and hyperparameters.
        
        Args:
            products_csv (str): Path to the products CSV file.
            images_csv (str): Path to the images CSV file.
            training_csv (str): Path where the processed training CSV will be saved.
            image_dir (str): Directory where the image files are stored.
            decoder_path (str): Path to save the decoder mapping as a pickle file.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of epochs to train.
            learning_rate (float): Learning rate for the optimizer.
            momentum (float): Momentum for the SGD optimizer.
            num_workers (int): Number of workers for data loading.
        """
        self.products_csv = products_csv
        self.images_csv = images_csv
        self.training_csv = training_csv
        self.image_dir = image_dir
        self.decoder_path = decoder_path

        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers

        # Use GPU if available, otherwise CPU.
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Placeholders for later use.
        self.num_classes = None
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def process_data(self):
        """
        Process the raw product and images data to generate the training CSV file.
        
        This method uses the ProductLabeler class to:
          - Load and process data.
          - Extract root categories and assign numeric labels.
          - Merge images data.
          - Save the processed data to the specified CSV file.
        It also saves the decoder mapping (numeric label → category) as a pickle file.
        """
        print("Processing data using ProductLabeler...")
        product_labeler = ProductLabeler(
            products_file=self.products_csv,
            images_file=self.images_csv,
            output_file=self.training_csv
        )
        product_labeler.process()

        # Save the decoder mapping for future use.
        with open(self.decoder_path, "wb") as f:
            pickle.dump(product_labeler.decoder, f)
        print(f"Decoder saved to {self.decoder_path}")

        # Set the number of classes using the encoder mapping.
        self.num_classes = len(product_labeler.encoder)
        print(f"Number of classes: {self.num_classes}")

    def setup_dataloaders(self):
        """
        Set up DataLoaders for training and validation.
        
        The dataset is loaded using ImageDataset and then split into:
          - Training set (70% of the data)
          - Validation set (15% of the data)
          - Test set (15% of the data; not used in this training loop)
        """
        print("Setting up DataLoaders...")
        dataset = ImageDataset(self.training_csv, self.image_dir)
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size  # Not used in training here

        train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=self.num_workers)
        print("DataLoaders are set up.")

    def setup_model(self):
        """
        Configure the pre-trained ResNet-50 model for transfer learning.
        
        The final fully connected layer is replaced with a new linear layer whose output size equals the number of classes.
        In this updated version, we first freeze all layers, then unfreeze the last two layers (layer4 and fc)
        so that the high-level features are fine-tuned for our dataset.
        The model is then moved to the appropriate device.
        """
        print("Setting up the model...")
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Freeze all parameters.
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two layers: layer4 and the fully connected layer.
        for param in self.model.layer4.parameters():
            param.requires_grad = True
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model = self.model.to(self.device)
        print("Model is set up with last two layers unfrozen and moved to device:", self.device)

    def train_model(self):
        """
        Train the model using a standard training loop with validation.
        At the end of every epoch, the model weights and metrics are saved.
        
        For each epoch:
          - The model is set to training mode and iterates over the training DataLoader.
          - For each batch, the loss is computed and the model parameters are updated.
          - The training loss is printed per batch.
          - After completing the epoch, the model is evaluated on the validation set and the average validation loss is printed.
          - The model weights are saved in a folder structure under "model_evaluation/<timestamp>/weights"
            with filenames indicating the epoch.
          - Metrics for the epoch (training loss, validation loss) are saved to a metrics file.
        """
        print("Starting training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), 
                              lr=self.learning_rate, momentum=self.momentum)

        # Create a folder for model evaluation that includes a timestamp in the folder name.
        base_dir = "model_evaluation"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        weights_dir = os.path.join(model_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        metrics_file = os.path.join(model_dir, "metrics.txt")
        
        # Open metrics file and write header.
        with open(metrics_file, "w") as f_metrics:
            f_metrics.write("Epoch,Training Loss,Validation Loss\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.model.train()  # Enable training mode
            running_loss = 0.0

            # Training loop over batches.
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()  # Clear previous gradients

                outputs = self.model(inputs)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss

                loss.backward()  # Backward pass (compute gradients)
                optimizer.step()  # Update model parameters

                running_loss += loss.item()
                print(f"Batch {batch_idx+1}/{len(self.train_loader)} - Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")

            # Validation phase: evaluate the model on the validation set.
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(self.val_loader)
            print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")
            
            # Save model weights for this epoch.
            weights_filename = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), weights_filename)
            print(f"Saved model weights to {weights_filename}")
            
            # Save metrics for this epoch.
            with open(metrics_file, "a") as f_metrics:
                f_metrics.write(f"{epoch+1},{avg_loss:.4f},{avg_val_loss:.4f}\n")
        
        print("Training complete.")

    def run(self):
        """
        Execute the full training pipeline:
          1. Process data and create the training CSV.
          2. Set up DataLoaders.
          3. Configure the model.
          4. Train the model (saving weights and metrics at the end of each epoch).
        """
        self.process_data()
        self.setup_dataloaders()
        self.setup_model()
        self.train_model()