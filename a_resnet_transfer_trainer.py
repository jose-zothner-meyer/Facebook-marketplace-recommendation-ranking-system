"""
a_resnet_transfer_trainer.py

This module defines the ResNetTransferLearner class, which encapsulates the full training pipeline:
  1. Processes raw product and image data (using ProductLabeler) to generate a training CSV.
  2. Creates encoder/decoder mappings and saves the decoder as "image_decoder.pkl".
  3. Sets up DataLoaders by splitting the dataset into training and validation sets.
  4. Configures a modified pretrained ResNet-50 model (via FineTunedResNet) by freezing all layers,
     then unfreezing the last two layers and replacing the classifier.
  5. Trains the model using a standard training loop with validation.
     At the end of every epoch, model weights and training/validation metrics are saved.

This design allows us to perform transfer learning and later extract features using the trained weights.
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
    FineTunedResNet is a modified version of the pretrained ResNet-50 model for transfer learning.
    
    The modifications include:
      - Loading a pretrained ResNet-50.
      - Freezing all parameters initially to preserve learned low-level features.
      - Unfreezing the parameters in the last block (layer4) and the original fully connected (fc) layer
        so that high-level features can be fine-tuned on our dataset.
      - Removing the original classification head (fc) and replacing it with a new classifier
        (wrapped in a nn.Sequential) that outputs the desired number of classes.
      - Combining the feature extractor (all layers except the original fc) with the new classifier
        into a single nn.Sequential model (stored as self.combined_model).
    """
    def __init__(self, num_classes: int) -> None:
        """
        Initialize the FineTunedResNet model.
        
        Args:
            num_classes (int): The number of output classes for the classifier.
        """
        super(FineTunedResNet, self).__init__()
        
        # Load a pretrained ResNet-50 model.
        resnet = models.resnet50(pretrained=True)
        
        # Freeze all layers to keep their pretrained weights unchanged.
        for param in resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze the parameters in the last convolutional block (layer4) and the final fc layer.
        # This allows these high-level layers to adapt to our specific dataset.
        for param in resnet.layer4.parameters():
            param.requires_grad = True
        for param in resnet.fc.parameters():
            param.requires_grad = True
        
        # Extract the feature extraction layers (all layers except the original fully connected layer).
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Define the new classifier layers. These layers include:
        #   - Adaptive average pooling to ensure a fixed spatial dimension,
        #   - Flattening of the output,
        #   - Dropout for regularization,
        #   - A linear layer that outputs 'num_classes' scores.
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(resnet.fc.in_features, num_classes)
        )
        
        # Combine the feature extractor and the new classifier into a single sequential model.
        # This combined model is used for the forward pass.
        self.combined_model = nn.Sequential(
            self.features,
            self.classifier
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the model.
        
        The input tensor is passed through the combined model, which consists of the original
        feature extractor and the new classifier.
        
        Args:
            x (torch.Tensor): Input image tensor.
            
        Returns:
            torch.Tensor: Output predictions (logits) for each class.
        """
        return self.combined_model(x)


class ResNetTransferLearner:
    """
    ResNetTransferLearner encapsulates the full training pipeline for the transfer learning model.
    
    This class handles:
      1. Data processing: Uses ProductLabeler to process raw CSV files and generate a training CSV,
         as well as to create encoder/decoder mappings (saved to "image_decoder.pkl").
      2. Data loading: Sets up training and validation DataLoaders by splitting the dataset.
      3. Model configuration: Instantiates the FineTunedResNet model (with unfrozen high-level layers)
         and moves it to the appropriate device.
      4. Training: Executes a training loop over a specified number of epochs, saving the model weights
         and training/validation metrics after each epoch.
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
        # File and folder paths.
        self.products_csv: str = products_csv
        self.images_csv: str = images_csv
        self.training_csv: str = training_csv
        self.image_dir: str = image_dir
        self.decoder_path: str = decoder_path
        
        # Training hyperparameters.
        self.batch_size: int = batch_size
        self.num_epochs: int = num_epochs
        self.learning_rate: float = learning_rate
        self.momentum: float = momentum
        self.num_workers: int = num_workers

        # Determine the device to use (GPU if available, else CPU).
        self.device: torch.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Placeholders for later components.
        self.num_classes: int = 0
        self.model: nn.Module = None  # This will store our FineTunedResNet model.
        self.train_loader: Any = None  # DataLoader for training data.
        self.val_loader: Any = None  # DataLoader for validation data.

    def process_data(self) -> None:
        """
        Process raw product and image data to create the training CSV file and save the decoder mapping.
        
        This function uses the ProductLabeler to:
          - Load and process the raw CSV files.
          - Extract the root categories from product data.
          - Create encoder and decoder mappings (category to numeric label and vice versa).
          - Merge the images data with the product data.
          - Save the final processed data to a CSV file.
        
        The decoder mapping is also saved as "image_decoder.pkl" for future inference.
        """
        print("Processing data using ProductLabeler...")
        product_labeler = ProductLabeler(
            products_file=self.products_csv,
            images_file=self.images_csv,
            output_file=self.training_csv
        )
        product_labeler.process()
        
        # Save the decoder mapping for later use.
        with open(self.decoder_path, "wb") as file:
            pickle.dump(product_labeler.decoder, file)
        print(f"Decoder mapping saved to {self.decoder_path}")
        
        # Update the number of classes based on the encoder mapping.
        self.num_classes = len(product_labeler.encoder)
        print(f"Number of classes: {self.num_classes}")

    def setup_dataloaders(self) -> None:
        """
        Set up the DataLoaders for training and validation.
        
        This method loads the dataset using the ImageDataset class and splits it into:
          - Training set: 70% of the data.
          - Validation set: 15% of the data.
          - Test set: Remaining 15% (not used in this training pipeline).
        """
        print("Setting up DataLoaders...")
        dataset = ImageDataset(self.training_csv, self.image_dir)
        total_size: int = len(dataset)
        train_size: int = int(0.7 * total_size)
        val_size: int = int(0.15 * total_size)
        test_size: int = total_size - train_size - val_size  # This portion is ignored.
        
        # Split the dataset into training, validation, and test sets.
        train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
        
        # Create DataLoaders for the training and validation datasets.
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, 
                                       shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, 
                                     shuffle=False, num_workers=self.num_workers)
        print("DataLoaders setup complete.")

    def setup_model(self) -> None:
        """
        Configure the FineTunedResNet model and move it to the appropriate device.
        
        This method instantiates the FineTunedResNet using the determined number of classes,
        then transfers the model to the GPU (if available) or CPU.
        """
        print("Setting up the FineTunedResNet model...")
        self.model = FineTunedResNet(self.num_classes)
        self.model.to(self.device)
        print("Model is set up on device:", self.device)

    def train_model(self) -> None:
        """
        Train the model using a standard training loop with validation.
        
        For each epoch:
          - The model is set to training mode and iterates over the training DataLoader.
          - For each batch, the loss is computed and backpropagated.
          - Model parameters (only for unfrozen layers) are updated via the optimizer.
          - Training loss is accumulated and printed.
          - After each epoch, the model is evaluated on the validation set, and the average validation loss is printed.
          - The model's weights are saved in a folder structure under "model_evaluation/<timestamp>/weights"
            with filenames indicating the epoch.
          - Training and validation metrics for each epoch are appended to a metrics file.
        """
        print("Starting training...")
        criterion = nn.CrossEntropyLoss()
        # Only parameters with requires_grad=True are passed to the optimizer.
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()),
                              lr=self.learning_rate, momentum=self.momentum)
        
        # Create a base directory to store model evaluation outputs.
        base_dir = "model_evaluation"
        os.makedirs(base_dir, exist_ok=True)
        
        # Create a timestamped folder for this training run.
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = os.path.join(base_dir, f"model_{timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create a subfolder for saving model weights.
        weights_dir = os.path.join(model_dir, "weights")
        os.makedirs(weights_dir, exist_ok=True)
        
        # Create a metrics file to log training and validation loss.
        metrics_file = os.path.join(model_dir, "metrics.txt")
        with open(metrics_file, "w") as f_metrics:
            f_metrics.write("Epoch,Training Loss,Validation Loss\n")
        
        # Loop over the number of epochs.
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            self.model.train()  # Set the model to training mode.
            running_loss: float = 0.0  # Accumulate training loss.
            
            # Iterate over batches from the training DataLoader.
            for batch_idx, (inputs, labels) in enumerate(self.train_loader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()  # Reset gradients.
                outputs = self.model(inputs)  # Forward pass.
                loss = criterion(outputs, labels)  # Compute loss.
                loss.backward()  # Backward pass.
                optimizer.step()  # Update parameters.
                
                running_loss += loss.item()
                print(f"Batch {batch_idx+1}/{len(self.train_loader)} - Loss: {loss.item():.4f}")
            
            # Compute and print the average training loss for the epoch.
            avg_train_loss: float = running_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")
            
            # Evaluate the model on the validation set.
            self.model.eval()  # Set the model to evaluation mode.
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
            
            # Save model weights at the end of the epoch.
            weights_filename = os.path.join(weights_dir, f"epoch_{epoch+1}.pth")
            torch.save(self.model.state_dict(), weights_filename)
            print(f"Saved model weights to {weights_filename}")
            
            # Append epoch metrics (training and validation loss) to the metrics file.
            with open(metrics_file, "a") as f_metrics:
                f_metrics.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f}\n")
        
        print("Training complete.")

    def run(self) -> None:
        """
        Execute the full training pipeline:
          1. Process raw product and image data to create the training CSV and save the decoder.
          2. Set up the training and validation DataLoaders.
          3. Configure the FineTunedResNet model.
          4. Train the model and save weights/metrics after each epoch.
        """
        self.process_data()
        self.setup_dataloaders()
        self.setup_model()
        self.train_model()


if __name__ == "__main__":
    # Create an instance of ResNetTransferLearner and run the training pipeline.
    learner = ResNetTransferLearner()
    learner.run()