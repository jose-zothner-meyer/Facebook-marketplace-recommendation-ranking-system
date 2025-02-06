"""
feature_extractor.py

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
    """
    Loads a trained classification model, removes its original classification head,
    and attaches a new fully connected layer that outputs a 1000-dimensional feature vector.
    
    The final feature extraction model is set to evaluation mode and its state dictionary is saved
    in a folder called 'final_model' with the filename 'image_model.pt'.
    """
    # Option 1: Load your trained model checkpoint.
    # Uncomment and modify these lines to load your checkpoint.
    #
    # model = models.resnet50(pretrained=False)
    # checkpoint_path = "model_evaluation/20250205_175327/weights/epoch_4.pth"
    # model.load_state_dict(torch.load(checkpoint_path))
    #
    # Option 2: For demonstration, we use a standard pretrained ResNet-50.
    model = models.resnet50(pretrained=True)
    
    # Get the number of input features for the original fc layer.
    num_ftrs = model.fc.in_features
    
    # Remove the original classification head by taking all layers except the final fc.
    # Then create a new Sequential that flattens the output and applies a new fc layer.
    feature_extractor = nn.Sequential(
        *list(model.children())[:-1],  # All layers except the original fc.
        nn.Flatten(),                   # Flatten the output tensor.
        nn.Linear(num_ftrs, 1000)         # New fully connected layer that outputs 1000 features.
    )
    
    # Set the model to evaluation mode.
    feature_extractor.eval()
    
    # Create a folder called 'final_model' if it does not exist.
    final_model_dir = "final_model"
    if not os.path.exists(final_model_dir):
        os.makedirs(final_model_dir)
    
    # Save the feature extraction model's state dictionary.
    save_path = os.path.join(final_model_dir, "image_model.pt")
    torch.save(feature_extractor.state_dict(), save_path)
    print(f"Feature extraction model saved to {save_path}")

if __name__ == "__main__":
    create_feature_extractor()