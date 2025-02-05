"""
create_feature_extractor.py

This script converts a trained classification model into a feature extraction model.
It removes the final classification head from the model and replaces it with a new fully connected layer
that outputs a 1000-dimensional feature vector. This feature extraction model can later be used with FAISS
for image similarity search.

After conversion, the final model weights are saved in a folder called 'final_model'
(within your project root) with the filename 'image_model.pt'.
"""

import os
import torch
import torch.nn as nn
from torchvision import models

def create_feature_extractor():
    # Load the trained classification model.
    # For demonstration purposes, we instantiate a pretrained ResNet-50.
    # If you have a specific checkpoint, uncomment and modify the following lines:
    #
    # model = models.resnet50(pretrained=False)
    # checkpoint_path = "path_to_your_trained_classification_model.pth"
    # model.load_state_dict(torch.load(checkpoint_path))
    #
    # Otherwise, use the standard pretrained model:
    model = models.resnet50(pretrained=True)
    
    # Get the number of input features to the final fully connected layer.
    num_ftrs = model.fc.in_features
    
    # Replace the classification head with a new fully connected layer that has 1000 neurons.
    # This new layer serves as the feature extraction head.
    model.fc = nn.Linear(num_ftrs, 1000)
    
    # (Optional) Set the model to evaluation mode.
    model.eval()
    
    # Create a new folder called 'final_model' (if it doesn't already exist).
    final_model_dir = "final_model"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    # Save the final model weights.
    save_path = os.path.join(final_model_dir, "image_model.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Feature extraction model saved to {save_path}")

if __name__ == "__main__":
    create_feature_extractor()
