"""
a_resnet_transfer_trainer.py

This module defines the ResNetTransferLearner class, which encapsulates the full training pipeline.
Key differences from your initial code:
  - The FineTunedResNet model now directly replaces the original fc layer (avoiding double pooling).
  - Only layer4 and the fc layers are unfrozen for fine-tuning.
Inline comments explain these modifications.
"""

import os
import pickle
import datetime
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models

from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset

class FineTunedResNet(nn.Module):
    """
    Modified ResNet-50 model for transfer learning.
    
    Key changes:
      - Direct replacement of the fc layer with a sequential block instead of creating a separate combined model.
      - This avoids applying AdaptiveAvgPool2d twice.
      - Only layer4 and fc layers are unfrozen.
    """
    def __init__(self, num_classes: int) -> None:
        super(FineTunedResNet, self).__init__()
        
        # Load pretrained ResNet-50.
        self.model = models.resnet50(pretrained=True)
        
        # Freeze all layers initially.
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters only in layer4 and the fc layer.
        for name, layer in self.model.named_children():
            if name in ['layer4', 'fc']:
                for param in layer.parameters():
                    param.requires_grad = True
        
        # Replace the fc layer with a sequential block.
        self.new_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(1000, num_classes)  # Final layer outputs class scores.
        )
        self.combined_model = nn.Sequential(
            self.model,
            self.new_layers
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using the modified ResNet-50 model."""
        return self.combined_model(x)


class ResNetTransferLearner:
    """
    Encapsulates the full training pipeline for the transfer learning model.
    
    Differences:
      - Uses the updated FineTunedResNet for a simpler model architecture.
      - Data loading and training loops are similar to the initial design.
    """
    def __init__(
        self,
        products_csv: str = "data/Cleaned_Products.csv",
        images_csv: str = "data/Images.csv",
        training_csv: str = "data/training_data.csv",
        image_dir: str = "cleaned_images/",
        decoder_path: str = "image_decoder.pkl",
        batch_size: int = 32,
        num_epochs: int = 5,
        learning_rate: float = 0.001,
        momentum: float = 0.9,
        num_workers: int = 4,
    ) -> None:
        self.products_csv: str = products_csv
        self.images_csv: str = images_csv
        self.training_csv: str = training_csv
        self.image_dir: str = image_dir
        self.decoder_path: str = decoder_path
        
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.num_workers: int = num_workers

        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.num_classes: int = 0
        self.model: nn.Module = None
        self.train_loader: Any = None
        self.val_loader: Any = None

    def process_data(self) -> None:
        """
        Process raw product and image data to create the training CSV and save the decoder mapping.
        """
        print("Processing data using ProductLabeler...")
        product_labeler = ProductLabeler(
            products_file=self.products_csv,
            images_file=self.images_csv,
            output_file=self.training_csv
        )
        product_labeler.process()
        
        with open(self.decoder_path, "wb") as file:
            pickle.dump(product_labeler.decoder, file)
        print(f"Decoder mapping saved to {self.decoder_path}")
        
        self.num_classes = len(product_labeler.encoder)
        print(f"Number of classes: {self.num_classes}")

    def setup_dataloaders(self) -> None:
        """
        Sets up DataLoaders for training and validation by splitting the dataset.
        """
        print("Setting up DataLoaders...")
        dataset = ImageDataset(self.training_csv, self.image_dir)
        total_size: int = len(dataset)
        train_size: int = int(0.7 * total_size)
        val_size: int = int(0.15 * total_size)
        test_size: int = total_size - train_size - val_size
        
        train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=self.num_workers)
        print("DataLoaders setup complete.")

    def setup_model(self) -> None:
        """
        Configures the FineTunedResNet model and moves it to the appropriate device.
        """
        print("Setting up the FineTunedResNet model...")
        self.model = FineTunedResNet(self.num_classes)
        self.model.to(self.device)
        print("Model is set up on device:", self.device)

    def train_model(self) -> None:
        """
        Trains the model using a standard training loop with validation.
        Saves model weights and metrics after each epoch.
        """
        print("Starting training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.learning_rate, momentum=self.momentum)
        
        base_dir = "model_evaluation"
        os.makedirs(base_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        weights_dir = os.path.join(model_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        metrics_file = os.path.join(model_dir, "metrics.txt")
        with open(metrics_file, "w") as f_metrics:
            f_metrics.write("Epoch,Training Loss,Validation Loss\n")
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.model.train()
            running_loss: float = 0.0
            
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                print(f"Batch {batch_idx+1}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
            
            avg_train_loss: float = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
            
            self.model.eval()
            running_val_loss: float = 0.0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    running_val_loss += loss.item()
            avg_val_loss: float = running_val_loss / len(self.val_loader)
            print(f"Epoch {epoch+1} Validation Loss: {avg_val_loss:.4f}")
            
            weights_filename = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), weights_filename)
            print(f"Saved model weights to {weights_filename}")
            
            with open(metrics_file, "a") as f_metrics:
                f_metrics.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
        
        print("Training complete.")

    def run(self) -> None:
        """
        Executes the full training pipeline: processes data, sets up DataLoaders,
        configures the model, and trains it.
        """
        self.process_data()
        self.setup_dataloaders()
        self.setup_model()
        self.train_model()


if __name__ == "__main__":
    learner = ResNetTransferLearner()
    learner.run()