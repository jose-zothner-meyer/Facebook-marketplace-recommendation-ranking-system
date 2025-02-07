"""
b_extract_embeddings.py

This script uses the feature extraction model to extract image embeddings for every valid image
in the 'cleaned_images' folder. For each image, it computes a 1000-dimensional feature vector using
the FeatureExtractionCNN model. The image identifier (derived from the filename without extension) is
used as the key in a dictionary, and the corresponding feature vector is stored as a list.
The resulting dictionary is saved as a JSON file named 'image_embeddings.json'.

Usage:
    python b_extract_embeddings.py
"""

import os
import json
import sys
from typing import Dict, Optional

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Import the feature extraction model and the transformation pipeline.
# The FeatureExtractionCNN model is defined in b_feature_extractor_model.py.
from b_feature_extractor_model import FeatureExtractionCNN, TRANSFORM_PIPELINE


def process_image(image_path: str) -> Optional[torch.Tensor]:
    """
    Load an image from disk, apply the predefined transformation pipeline, and add a batch dimension.

    The transformation pipeline (TRANSFORM_PIPELINE) resizes the image, converts it to a tensor,
    and normalizes it using ImageNet statistics. The unsqueeze operation adds a batch dimension,
    resulting in a tensor of shape (1, 3, 256, 256), which is required for the model input.

    Args:
        image_path (str): The path to the image file.

    Returns:
        Optional[torch.Tensor]: The processed image tensor of shape (1, 3, 256, 256), or None if an error occurs.
    """
    try:
        # Open the image file and convert it to RGB.
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        # If there is an error (e.g., file is corrupted or not an image), print an error message and return None.
        print(f"Error opening image {image_path}: {e}")
        return None

    # Apply the transformation pipeline to prepare the image for the model.
    img_tensor = TRANSFORM_PIPELINE(image)
    # Add a batch dimension so the tensor shape becomes (1, 3, 256, 256).
    return img_tensor.unsqueeze(0)


def extract_all_embeddings(image_folder: str, model: FeatureExtractionCNN) -> Dict[str, list]:
    """
    Iterate over all valid images in the specified folder, extract their embeddings using the provided model,
    and store the embeddings in a dictionary.

    For each image:
      - The image is loaded and preprocessed using the process_image function.
      - The model computes a 1000-dimensional embedding vector.
      - The image ID (filename without extension) is used as the dictionary key.
      - The embedding vector is converted to a list (from a PyTorch tensor) for JSON serialization.

    Args:
        image_folder (str): The directory containing images.
        model (FeatureExtractionCNN): The feature extraction model to use for computing embeddings.

    Returns:
        Dict[str, list]: A dictionary mapping image IDs to their corresponding embedding vectors (as lists).
    """
    # Define acceptable file extensions for images.
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # Get a list of all files in the folder that have a valid image extension.
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    
    # If no valid images are found, print a message and exit the program.
    if not image_files:
        print(f"No valid image files found in '{image_folder}'.")
        sys.exit(1)

    embeddings_dict: Dict[str, list] = {}

    # Process images in sorted order (to ensure consistent ordering).
    for image_filename in sorted(image_files):
        image_path = os.path.join(image_folder, image_filename)
        # Process the image; if processing fails, skip this image.
        img_tensor = process_image(image_path)
        if img_tensor is None:
            print(f"Skipping image {image_filename} due to processing error.")
            continue

        # Use the model to extract the embedding without computing gradients.
        with torch.no_grad():
            embedding_tensor = model(img_tensor)
        # Remove the batch dimension from the output (from shape (1, 1000) to (1000,)).
        embedding_tensor = embedding_tensor.squeeze(0)
        # Use the filename (without extension) as the image ID.
        image_id = os.path.splitext(image_filename)[0]
        # Convert the tensor to a list so it can be saved as JSON.
        embeddings_dict[image_id] = embedding_tensor.tolist()
        print(f"Processed image: {image_filename}")

    return embeddings_dict


def main() -> None:
    """
    Main function to extract image embeddings for all images in the 'cleaned_images' folder.

    The function performs the following steps:
      1. Initialize the feature extraction model and load saved weights from "final_model/image_model.pt" if available.
      2. Process every valid image in the 'cleaned_images' folder using the FeatureExtractionCNN model.
      3. Build a dictionary mapping image IDs (derived from filenames without extension) to their embedding vectors.
      4. Save the embeddings dictionary as a JSON file named "image_embeddings.json".
    """
    image_folder: str = "cleaned_images"
    
    # Instantiate the feature extraction model.
    model = FeatureExtractionCNN()
    model_path: str = "final_model/image_model.pt"
    
    # If a trained model weights file exists, load it; otherwise, use the model with default weights.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded saved model weights.")
    else:
        print("Model weights not found. Using default parameters.")
    # Set the model to evaluation mode to disable dropout and other training-specific behavior.
    model.eval()

    # Extract embeddings for all valid images in the folder.
    embeddings_dict = extract_all_embeddings(image_folder, model)
    
    # Save the dictionary of embeddings to a JSON file.
    output_json: str = "image_embeddings.json"
    with open(output_json, "w") as json_file:
        json.dump(embeddings_dict, json_file)
    print(f"Saved embeddings for {len(embeddings_dict)} images to {output_json}")


if __name__ == "__main__":
    main()