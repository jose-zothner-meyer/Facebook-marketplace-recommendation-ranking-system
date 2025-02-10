"""
train_tensorboard.py

This module defines the train function that uses TensorBoard to log training and validation loss.
It splits the dataset into training, validation, and test sets, then trains the provided model,
logging metrics along the way.

Key differences:
  - Uses ImageDataset to load data.
  - Splits data into 70% train, 15% validation, and 15% test.
  - Logs metrics to TensorBoard for visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torch.utils.tensorboard import SummaryWriter
from image_dataset_pytorch import ImageDataset

def train(model, *, epochs=5, log_dir="logs/experiment_1"):
    """
    Train the given model for a specified number of epochs with TensorBoard logging.
    
    Steps:
      - Load dataset using ImageDataset.
      - Split the dataset into training (70%), validation (15%), and test (15%).
      - Create DataLoaders for training and validation.
      - Run a training loop, logging losses to TensorBoard.
      - Evaluate on the validation set at the end of each epoch.
    
    Args:
        model: A PyTorch model to be trained.
        epochs (int, optional): Number of epochs (default 5).
        log_dir (str, optional): Directory for TensorBoard logs.
    """
    # Define file paths and hyperparameters.
    training_csv = "data/training_data.csv"  # Processed training CSV.
    image_dir = "cleaned_images/"            # Directory containing images.
    batch_size = 32                          # Batch size.

    # Initialize TensorBoard SummaryWriter.
    writer = SummaryWriter(log_dir=log_dir)
    
    # Load the full dataset.
    dataset = ImageDataset(training_csv, image_dir)
    total_size = len(dataset)
    
    # Define dataset split sizes.
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # Test set (unused here).
    
    # Split the dataset.
    train_dataset, val_dataset, _ = random_split(dataset, [train_size, val_size, test_size])
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device.
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Define loss and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    global_step = 0  # For TensorBoard logging.

    # Training loop.
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        model.train()  # Set model to training mode.
        running_loss = 0.0  # Accumulate loss.

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()  # Reset gradients.
            outputs = model(inputs)  # Forward pass.
            loss = criterion(outputs, labels)  # Compute loss.
            loss.backward()  # Backpropagate.
            optimizer.step()  # Update parameters.

            running_loss += loss.item()
            global_step += 1

            # Log batch loss.
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            print(f"Batch {batch_idx+1}/{len(train_loader)} - Training Loss: {loss.item():.4f}")

        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} Average Training Loss: {avg_train_loss:.4f}")

        # Validation.
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1} - Validation Loss: {avg_val_loss:.4f}")

        # Log validation loss.
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch+1)

    writer.close()
    print("Training complete. TensorBoard logs saved at:", log_dir)
