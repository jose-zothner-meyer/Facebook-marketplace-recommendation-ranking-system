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
from PIL import Image
from torchvision import transforms

from b_feature_extractor_model import TRANSFORM_PIPELINE


def process_image(image_path: str) -> Optional[torch.Tensor]:
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

    img_tensor = TRANSFORM_PIPELINE(image)
    return img_tensor.unsqueeze(0)


def main() -> None:
    """
    Main function to process the first image in the 'cleaned_images' folder.

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

    image_filename: str = sorted(image_files)[0]
    image_path: str = os.path.join(image_folder, image_filename)
    print(f"Processing first image: {image_filename}")

    processed_image = process_image(image_path)
    if processed_image is None:
        sys.exit(1)

    print(f"Processed image shape: {processed_image.shape}")


if __name__ == "__main__":
    main()