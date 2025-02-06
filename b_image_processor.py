"""
image_processor.py

This script processes the first valid image found in the 'cleaned_images' folder by applying the same
transformations used during training and adding a batch dimension (to convert the image from shape 
(n_channels, height, width) to (1, n_channels, height, width)). This prepares the image to be fed into
the feature extraction model.

Usage:
    python image_processor.py
"""

import os
import sys
from typing import Optional

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
      - Removes its final classification layer.
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


# Transformation pipeline must match the one used during training.
_TRANSFORM_PIPELINE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def _process_image(image_path: str) -> Optional[torch.Tensor]:
    """
    Loads an image, applies the transformation pipeline, and adds a batch dimension.

    Args:
        image_path (str): The file path of the image.

    Returns:
        Optional[torch.Tensor]: Processed image tensor of shape (1, 3, 256, 256), or None if an error occurs.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    img_tensor = _TRANSFORM_PIPELINE(image)
    # Add a batch dimension to convert shape from (3, 256, 256) to (1, 3, 256, 256)
    return img_tensor.unsqueeze(0)


def main() -> None:
    """
    Main function to process the first image found in the 'cleaned_images' folder.

    Steps:
      1. Identify the first valid image file in 'cleaned_images'.
      2. Process the image using the transformation pipeline.
      3. Print the shape of the processed image tensor.
    """
    image_folder: str = "cleaned_images"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print(f"No valid images found in '{image_folder}'.")
        sys.exit(1)

    # Automatically select the first image (alphabetically)
    image_filename: str = sorted(image_files)[0]
    image_path: str = os.path.join(image_folder, image_filename)
    print(f"Processing first image: {image_filename}")

    processed_image = _process_image(image_path)
    if processed_image is None:
        sys.exit(1)

    print(f"Processed image shape: {processed_image.shape}")


if __name__ == "__main__":
    main()