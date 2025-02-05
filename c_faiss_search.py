"""
faiss_search.py

This script loads a dictionary of image embeddings from a JSON file,
builds a FAISS index for vector search, and performs a search for similar images.

Requirements:
- The JSON file "image_embeddings.json" must exist and contain a dictionary
  mapping image IDs to their embedding vectors.
- The query image is automatically selected from the "cleaned_images" folder.
- Model weights are loaded from "final_model/image_model.pt" if needed.

Usage:
    python faiss_search.py
"""

import os
import json
from typing import List, Tuple

import numpy as np
import faiss  # Install via pip install faiss-cpu (or faiss-gpu)
import torch
from PIL import Image
from torchvision import transforms

from b_feature_extractor_model import FeatureExtractionCNN, TRANSFORM_PIPELINE


def process_image(image_path: str) -> torch.Tensor:
    """
    Processes an image: loads it, applies transformations, and adds a batch dimension.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: Processed image tensor of shape (1, 3, 256, 256).
    """
    image = Image.open(image_path).convert("RGB")
    img_tensor = TRANSFORM_PIPELINE(image)
    return img_tensor.unsqueeze(0)


def load_embeddings(json_file: str) -> Tuple[List[str], np.ndarray]:
    """
    Loads image embeddings from a JSON file.

    Args:
        json_file (str): Path to the JSON file containing embeddings.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing a list of image IDs and an array of embeddings.
    """
    with open(json_file, "r") as file:
        data = json.load(file)
    image_ids = list(data.keys())
    embeddings = [np.array(data[img_id], dtype=np.float32) for img_id in image_ids]
    embeddings_array = np.vstack(embeddings)
    return image_ids, embeddings_array


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index using L2 distance from the provided embeddings.

    Args:
        embeddings (np.ndarray): Array of shape (n, d) with n embeddings of dimension d.

    Returns:
        faiss.Index: FAISS index with embeddings added.
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index


def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
    """
    Searches the FAISS index for the k nearest neighbors of the query embedding.

    Args:
        index (faiss.Index): FAISS index.
        query_embedding (np.ndarray): Query embedding of shape (d,).
        k (int): Number of nearest neighbors to retrieve.

    Returns:
        Tuple[List[float], List[int]]: Distances and indices of the top k nearest neighbors.
    """
    query = np.expand_dims(query_embedding, axis=0)
    distances, indices = index.search(query, k)
    return distances[0].tolist(), indices[0].tolist()


def main() -> None:
    """
    Main function to:
      1. Load image embeddings from a JSON file.
      2. Build a FAISS index.
      3. Process a query image (first image in 'cleaned_images').
      4. Extract its embedding.
      5. Perform a FAISS search to find similar images.
      6. Print the top k similar image IDs and their distances.
    """
    json_file: str = "image_embeddings.json"
    if not os.path.exists(json_file):
        print(f"Embeddings file {json_file} not found.")
        return

    image_ids, embeddings = load_embeddings(json_file)
    print(f"Loaded embeddings for {len(image_ids)} images.")

    index = build_faiss_index(embeddings)
    print("FAISS index built successfully.")

    image_folder: str = "cleaned_images"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    if not image_files:
        print(f"No valid images found in '{image_folder}'.")
        return
    query_image_filename = sorted(image_files)[0]
    query_image_path = os.path.join(image_folder, query_image_filename)
    print(f"Using query image: {query_image_filename}")

    img_tensor = process_image(query_image_path)
    model = FeatureExtractionCNN()
    model_path: str = "final_model/image_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded model weights for query image extraction.")
    else:
        print("Model weights not found; using default parameters.")
    model.eval()
    with torch.no_grad():
        query_embedding = model(img_tensor).squeeze(0).numpy()

    k: int = 5
    distances, indices = search_faiss(index, query_embedding, k)
    print(f"Top {k} similar images:")
    for dist, idx in zip(distances, indices):
        print(f"Image ID: {image_ids[idx]}, Distance: {dist}")


if __name__ == "__main__":
    main()