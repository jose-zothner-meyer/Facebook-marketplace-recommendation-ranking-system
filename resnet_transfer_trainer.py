"""
resnet_transfer_trainer.py

This module defines the ResNetTransferLearner class, which leverages transfer learning
to fine tune a pre-trained ResNet-50 model on a custom image dataset. The class handles:
  - Processing product and image data to generate a training CSV file.
  - Saving the decoder mapping (which converts numeric labels back to original categories).
  - Setting up the DataLoader.
  - Configuring and fine tuning the ResNet-50 model.
  - Training the model.

The decoder mapping is saved as "image_decoder.pkl" for future inference.
"""

import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset

class ResNetTransferLearner:
    """
    A class for performing transfer learning using a pre-trained ResNet-50 model.
    
    This class processes product and image data, fine tunes the ResNet-50 model for image 
    classification, and saves the decoder mapping for converting numeric labels back to category names.
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
            training_csv (str): Path where the processed training data CSV will be saved.
            image_dir (str): Directory where the image files are stored.
            decoder_path (str): Path where the decoder mapping (pickle file) will be saved.
            batch_size (int): Number of samples per batch for training.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.
            momentum (float): Momentum factor for the SGD optimizer.
            num_workers (int): Number of subprocesses for data loading.
        """
        # File paths for input and output data
        self.products_csv = products_csv
        self.images_csv = images_csv
        self.training_csv = training_csv
        self.image_dir = image_dir
        self.decoder_path = decoder_path

        # Hyperparameters for training
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.num_workers = num_workers

        # Device configuration: use GPU if available, otherwise CPU
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Placeholders for DataLoader, model, and the number of classes
        self.dataloader = None
        self.model = None
        self.num_classes = None

    def process_data(self):
        """
        Process the product and image data to create the training dataset.

        This method leverages the ProductLabeler class to process the CSV files and generate 
        a training CSV file. It also creates encoder/decoder mappings for categories. The 
        decoder mapping (numeric label to original category) is saved as a pickle file.
        """
        # Initialize ProductLabeler with provided file paths
        product_labeler = ProductLabeler(
            products_file=self.products_csv,
            images_file=self.images_csv,
            output_file=self.training_csv
        )
        # Run the full processing pipeline (load data, extract categories, encode, merge, etc.)
        product_labeler.process()

        # Save the decoder mapping (for converting numeric labels back to original category names)
        with open(self.decoder_path, "wb") as f:
            pickle.dump(product_labeler.decoder, f)
        print(f"Decoder saved to {self.decoder_path}")

        # Determine the number of classes from the encoder mapping
        self.num_classes = len(product_labeler.encoder)
        print(f"Number of classes: {self.num_classes}")

    def setup_dataloader(self):
        """
        Set up the DataLoader for training using the ImageDataset.

        The DataLoader loads images and labels from the training CSV file, applying any necessary 
        transformations as defined in the ImageDataset class.
        """
        # Create an instance of the custom ImageDataset
        dataset = ImageDataset(self.training_csv, self.image_dir)

        # Initialize the DataLoader with the dataset, batch size, shuffling, and number of workers
        self.dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
        print("DataLoader is set up.")

    def setup_model(self):
        """
        Set up the pre-trained ResNet-50 model for transfer learning.

        The final fully connected layer is replaced with a new linear layer whose output size is 
        equal to the number of classes in the dataset.
        """
        # Load the pre-trained ResNet-50 model from torchvision
        self.model = models.resnet50(pretrained=True)
        
        # Get the number of input features of the original final (fully connected) layer
        num_ftrs = self.model.fc.in_features
        
        # Replace the final fully connected layer with a new one with output size = num_classes
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Move the model to the appropriate device (GPU or CPU)
        self.model = self.model.to(self.device)
        print("Model is set up and moved to device:", self.device)

    def train(self):
        """
        Train the ResNet-50 model using the prepared DataLoader.

        This method runs the training loop for the specified number of epochs. For each batch, it:
          - Moves data to the appropriate device.
          - Performs a forward pass.
          - Computes the loss using CrossEntropyLoss.
          - Performs backpropagation and updates the model weights.
          - Prints the average loss after each epoch.
        """
        # Define the loss function (cross-entropy for multi-class classification)
        criterion = nn.CrossEntropyLoss()
        # Define the optimizer (SGD with specified learning rate and momentum)
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=self.momentum)

        # Training loop for each epoch
        for epoch in range(self.num_epochs):
            self.model.train()  # Set the model to training mode
            running_loss = 0.0  # Initialize running loss for the epoch

            # Iterate over the DataLoader to fetch batches of images and labels
            for inputs, labels in self.dataloader:
                # Move inputs and labels to the configured device
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Zero the gradients for the optimizer to avoid accumulation
                optimizer.zero_grad()

                # Forward pass: compute model predictions
                outputs = self.model(inputs)
                # Compute the loss between predictions and true labels
                loss = criterion(outputs, labels)

                # Backward pass: compute gradients of loss with respect to model parameters
                loss.backward()
                # Update model parameters using the optimizer
                optimizer.step()

                # Accumulate the loss (scaled by the batch size)
                running_loss += loss.item() * inputs.size(0)
            
            # Calculate the average loss for the epoch
            epoch_loss = running_loss / len(self.dataloader.dataset)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

        print("Training complete.")

    def run(self):
        """
        Execute the full training pipeline:
          1. Process data and generate training CSV along with encoder/decoder mappings.
          2. Set up the DataLoader for the dataset.
          3. Configure the pre-trained model for transfer learning.
          4. Train the model.
        """
        self.process_data()      # Process and prepare the dataset
        self.setup_dataloader()  # Set up the DataLoader for iterating over the dataset
        self.setup_model()       # Configure the pre-trained ResNet-50 model for fine tuning
        self.train()             # Train the model on the dataset

# End of ResNetTransferLearner class
