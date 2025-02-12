"""
b_extract_embeddings.py

This file contains the integrated training and feature extraction pipeline.
It combines data processing, model training (with additional accuracy metrics), 
model conversion, and image embedding extraction into one function: run_pipeline().

This script has been updated to include additional procedures from Jose_sandbox.ipynb.

Steps:
1. Data Processing: Reads image labels and prepares training, validation, and test sets.
2. Model Training: Fine-tunes a ResNet-based model for image classification.
3. Model Conversion: Converts the trained model to extract image embeddings.
4. Embedding Extraction: Generates and saves feature embeddings for images.
"""

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

# Import the FineTunedResNet model from the teacher's file
from a_resnet_transfer_trainer import FineTunedResNet
# Import the custom dataset class
from image_dataset_pytorch import ImageDataset

# Set device for PyTorch computations (GPU if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters (adjust as needed)
num_classes = 13  # Number of unique labels/classes in the dataset
saved_weights = 'epoch_5.pth'  # Model weights file
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
