"""
image_processor.py

This script processes the first valid image found in the 'cleaned_images' folder by applying the transformation pipeline
and adding a batch dimension. This prepares the image for the feature extraction model.

Key differences from your initial code:
  - Inline comments are added for clarity.
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
    Load an image, apply the transformation pipeline, and add a batch dimension.
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
    Main function: processes the first valid image in 'cleaned_images' and prints its tensor shape.
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