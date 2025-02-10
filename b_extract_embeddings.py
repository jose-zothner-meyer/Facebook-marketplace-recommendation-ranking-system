"""
b_extract_embeddings.py

This script extracts image embeddings using the feature extraction model.
It loads the final model from 'data/final_model/image_model.pt' and uses the transformation
pipeline defined in b_feature_extractor_model.py.

Key differences from your initial code:
  - File paths now match the outputs from the integrated pipeline.
  - Inline comments clarify each step.
"""

import os
import json
import sys
from typing import Dict, Optional

import torch
from PIL import Image
from torchvision import transforms

from b_feature_extractor_model import FeatureExtractionCNN, TRANSFORM_PIPELINE

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

def extract_all_embeddings(image_folder: str, model: FeatureExtractionCNN) -> Dict[str, list]:
    """
    Iterate over valid images in the folder, extract embeddings using the model,
    and return a dictionary mapping image IDs to embedding vectors.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print(f"No valid image files found in '{image_folder}'.")
        sys.exit(1)
    embeddings_dict: Dict[str, list] = {}
    for image_filename in sorted(image_files):
        image_path = os.path.join(image_folder, image_filename)
        img_tensor = process_image(image_path)
        if img_tensor is None:
            print(f"Skipping image {image_filename} due to processing error.")
            continue
        with torch.no_grad():
            embedding_tensor = model(img_tensor)
        embedding_tensor = embedding_tensor.squeeze(0)
        image_id = os.path.splitext(image_filename)[0]
        embeddings_dict[image_id] = embedding_tensor.tolist()
        print(f"Processed image: {image_filename}")
    return embeddings_dict

def main() -> None:
    """
    Main function to extract image embeddings for all images in 'cleaned_images'.
    """
    image_folder: str = "cleaned_images"
    
    # Instantiate the feature extraction model.
    model = FeatureExtractionCNN()
    model_path: str = "data/final_model/image_model.pt"
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded saved model weights.")
    else:
        print("Model weights not found. Using default parameters.")
    model.eval()

    embeddings_dict = extract_all_embeddings(image_folder, model)
    
    output_json: str = "data/output/image_embeddings.json"
    with open(output_json, "w") as json_file:
        json.dump(embeddings_dict, json_file)
    print(f"Saved embeddings for {len(embeddings_dict)} images to {output_json}")

if __name__ == "__main__":
    main()
