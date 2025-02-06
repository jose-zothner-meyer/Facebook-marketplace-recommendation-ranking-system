"""
resnet_transfer_trainer.py

This module defines the ResNetTransferLearner class, which encapsulates the full training pipeline:
  1. Process raw product and image data (using ProductLabeler) to generate a training CSV.
  2. Create encoder/decoder mappings and save the decoder as "image_decoder.pkl".
  3. Set up DataLoaders by splitting the dataset into training and validation sets.
  4. Configure a modified pretrained ResNet-50 model (via FineTunedResNet) by freezing all layers,
     then unfreezing the last two layers and replacing the classifier.
  5. Train the model using a standard training loop with validation.
     At the end of every epoch, model weights and metrics are saved.
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

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the FineTunedResNet model.
        
        This constructor:
          - Loads a pretrained ResNet-50.
          - Freezes all parameters.
          - Unfreezes the last two layers (layer4 and fc) for fine-tuning.
          - Removes the original fc layer and replaces it with new classifier layers wrapped in a Sequential.
          - Combines the feature extractor (all layers except the original fc) with the new classifier in one Sequential module.
        
        Args:
            num_classes (int): The number of output classes.
        """
        super(FineTunedResNet, self).__init__()
        
        # Load the pretrained ResNet-50 model.
        resnet = models.resnet50(pretrained=True)
        
        # Freeze all parameters.
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze parameters in layer4 and fc.
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True
        
        # Extract the feature extractor part (all layers except the original fc).
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Define new classifier layers. Here we wrap our additional layers in a Sequential.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Ensure a fixed-size output.
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, num_classes)
        )
        
        # Combine the feature extractor and new classifier into one Sequential.
        self.combined_model = nn.Sequential(
            self.features,
            self.classifier
        )
    
    def forward(self, x):
        """
        Forward pass through the combined model.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Output predictions.
        """
        return self.combined_model(x)

class ResNetTransferLearner:
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
        Initialize the ResNetTransferLearner with file paths and training hyperparameters.
        
        Args:
            products_csv (str): Path to the products CSV file.
            images_csv (str): Path to the images CSV file.
            training_csv (str): Path where the processed training CSV will be saved.
            image_dir (str): Directory where image files are stored.
            decoder_path (str): Path to save the decoder mapping.
            batch_size (int): Batch size for training.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            momentum (float): Momentum for the SGD optimizer.
            num_workers (int): Number of worker processes for data loading.
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_classes = None
        self.model = None
        self.train_loader = None
        self.val_loader = None

    def process_data(self):
        """
        Process the raw product and image data to create the training CSV file.
        
        Uses ProductLabeler to:
          - Load and process the CSVs.
          - Extract root categories and assign numeric labels.
          - Merge image data.
          - Save the processed data.
        Also saves the decoder mapping as a pickle file.
        """
        print("Processing data using ProductLabeler...")
        product_labeler = ProductLabeler(
            products_file=self.products_csv,
            images_file=self.images_csv,
            output_file=self.training_csv
        )
        product_labeler.process()
        with open(self.decoder_path, "wb") as f:
            pickle.dump(product_labeler.decoder, f)
        print(f"Decoder saved to {self.decoder_path}")
        self.num_classes = len(product_labeler.encoder)
        print(f"Number of classes: {self.num_classes}")

    def setup_dataloaders(self):
        """
        Set up DataLoaders for training and validation.
        
        Splits the dataset (loaded via ImageDataset) into:
          - Training set (70%)
          - Validation set (15%)
          - Test set (15% - not used here)
        """
        print("Setting up DataLoaders...")
        dataset = ImageDataset(self.training_csv, self.image_dir)
        total_size = len(dataset)
        train_size = int(0.7 * total_size)
        val_size = int(0.15 * total_size)
        test_size = total_size - train_size - val_size  # Not used in training
        
        train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=self.num_workers)
        print("DataLoaders are set up.")

    def setup_model(self):
        """
        Configure the model using FineTunedResNet.
        
        The model is instantiated with the number of classes, moved to the appropriate device,
        and the final two layers (layer4 and fc) are unfrozen for fine-tuning.
        """
        print("Setting up the FineTunedResNet model...")
        self.model = FineTunedResNet(self.num_classes)
        self.model = self.model.to(self.device)
        print("Model is set up and moved to device:", self.device)

    def train_model(self):
        """
        Train the model using a standard training loop with validation.
        
        At the end of each epoch:
          - The model's weights are saved under "model_evaluation/<timestamp>/weights" 
            with filenames indicating the epoch.
          - Training and validation losses are saved to a metrics file.
        """
        print("Starting training...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.learning_rate, momentum=self.momentum)
        
        # Create folder structure for saving model evaluation outputs.
        base_dir = "model_evaluation"
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
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
            running_loss = 0.0
            
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
            
            avg_loss = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_loss:.4f}")
            
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
            
            # Save model weights.
            weights_filename = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), weights_filename)
            print(f"Saved model weights to {weights_filename}")
            
            # Save epoch metrics.
            with open(metrics_file, "a") as f_metrics:
                f_metrics.write(f"{epoch+1},{avg_loss:.4f},{avg_val_loss:.4f}\n")
        
        print("Training complete.")

    def run(self):
        """
        Execute the full training pipeline:
          1. Process data to create the training CSV and save the decoder.
          2. Set up DataLoaders.
          3. Configure the model (using FineTunedResNet).
          4. Train the model (saving weights and metrics each epoch).
        """
        self.process_data()
        self.setup_dataloaders()
        self.setup_model()
        self.train_model()

# Once training is complete, if you want to turn your model into a feature extractor, you can remove your additional classifier layers using:

# feature_extractor = torch.nn.Sequential(*list(model.combined_model.children())[:-1])

# This will give you a model that outputs the feature vector for each input image.