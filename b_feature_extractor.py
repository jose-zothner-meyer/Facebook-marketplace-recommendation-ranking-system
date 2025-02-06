"""
extract_embeddings.py

This script uses a feature extraction model (a modified ResNet50) to extract image embeddings
for every valid image found in the 'cleaned_images' folder. It creates a dictionary where each key is the image id
(derived from the filename without extension) and the value is the corresponding image embedding.
The dictionary is saved as a JSON file named 'image_embeddings.json'.

Usage:
    python extract_embeddings.py
"""

import os
import json
import sys
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image


class FeatureExtractionCNN(nn.Module):
    """
    FeatureExtractionCNN modifies a pretrained ResNet50 model to extract image embeddings.

    The model:
      - Loads a pretrained ResNet50.
      - Removes the final classification layer.
      - Adds a new fully connected layer mapping the 2048 features to a 1000-dimensional embedding.
    """
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer.
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        self.feature_fc = nn.Linear(2048, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract image embeddings.

        Args:
            images (torch.Tensor): Batch of images with shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Extracted embeddings with shape (batch_size, 1000).
        """
        features = self.resnet50(images)
        features = features.view(features.size(0), -1)
        features = self.feature_fc(features)
        return features


# Transformation pipeline must match training transformations.
_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _process_image(image_path: str) -> Optional[torch.Tensor]:
    """
    Load an image, apply transformations, and add a batch dimension.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Optional[torch.Tensor]: Processed image tensor (1, 3, 256, 256) or None on error.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    img_tensor = _TRANSFORM_PIPELINE(image)
    return img_tensor.unsqueeze(0)  # Add batch dimension


def extract_all_embeddings(image_folder: str, model: FeatureExtractionCNN) -> Dict[str, list]:
    """
    Loop over all valid images in a folder, extract their embeddings, and store them in a dictionary.

    Args:
        image_folder (str): Directory containing images.
        model (FeatureExtractionCNN): The feature extraction model.

    Returns:
        Dict[str, list]: A dictionary mapping image ids (filename without extension) to embedding lists.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid image files found in '{image_folder}'.")
        sys.exit(1)

    embeddings_dict: Dict[str, list] = {}

    # Process each image.
    for image_filename in sorted(image_files):
        image_path = os.path.join(image_folder, image_filename)
        img_tensor = _process_image(image_path)
        if img_tensor is None:
            print(f"Skipping image {image_filename} due to processing error.")
            continue

        with torch.no_grad():
            embedding_tensor = model(img_tensor)
        # Remove the batch dimension.
        embedding_tensor = embedding_tensor.squeeze(0)
        embeddings_dict[os.path.splitext(image_filename)[0]] = embedding_tensor.tolist()
        print(f"Processed image: {image_filename}")

    return embeddings_dict


def main() -> None:
    """
    Main function to extract image embeddings for every image in the 'cleaned_images' folder.

    Steps:
      1. Initialize the feature extraction model and load saved weights if available.
      2. Process every valid image in the 'cleaned_images' folder.
      3. Build a dictionary mapping image ids to their embeddings.
      4. Save the dictionary as 'image_embeddings.json'.
    """
    image_folder: str = "cleaned_images"
    model = FeatureExtractionCNN()
    model_path: str = "final_model/image_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded saved model weights.")
    else:
        print("Model weights not found. Using default model parameters.")
    model.eval()

    embeddings_dict = extract_all_embeddings(image_folder, model)
    output_json: str = "image_embeddings.json"
    with open(output_json, "w") as json_file:
        json.dump(embeddings_dict, json_file)
    print(f"Saved embeddings for {len(embeddings_dict)} images to {output_json}")


if __name__ == "__main__":
    main()