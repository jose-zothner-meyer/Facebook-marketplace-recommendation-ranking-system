"""
c_faiss_search.py

This script loads a dictionary of image embeddings from a JSON file,
builds a FAISS index for vector search, and performs a search for similar images.

Requirements:
  - The JSON file "image_embeddings.json" must exist and contain a dictionary
    mapping image IDs to their embedding vectors.
  - A query image is automatically selected from the "cleaned_images" folder.
  - The feature extraction model weights are loaded from "final_model/image_model.pt" if available.

Usage:
    python c_faiss_search.py
"""

import os
import json
from typing import List, Tuple

import numpy as np
import faiss  # FAISS is used for efficient similarity search (install via pip install faiss-cpu or faiss-gpu)
import torch
from PIL import Image

# Import the feature extraction model and transformation pipeline.
from b_feature_extractor_model import FeatureExtractionCNN, TRANSFORM_PIPELINE


def process_image(image_path: str) -> torch.Tensor:
    """
    Load an image from the given path, apply the predefined transformation pipeline,
    and add a batch dimension to make it compatible with model input.

    Args:
        image_path (str): Path to the image file.

    Returns:
        torch.Tensor: A processed image tensor with shape (1, 3, 256, 256).
                      The batch dimension is added using unsqueeze(0).
    """
    # Open the image file and convert it to RGB mode.
    image = Image.open(image_path).convert("RGB")
    # Apply the transformation pipeline (resizing, converting to tensor, normalization, etc.).
    img_tensor = TRANSFORM_PIPELINE(image)
    # Add an extra dimension at position 0 to create a batch of size 1.
    return img_tensor.unsqueeze(0)


def load_embeddings(json_file: str) -> Tuple[List[str], np.ndarray]:
    """
    Load image embeddings from a JSON file and convert them to a contiguous NumPy array.

    The JSON file is expected to contain a dictionary where:
      - The keys are image IDs (typically filenames without extension).
      - The values are embedding vectors (lists of floats).

    Args:
        json_file (str): Path to the JSON file containing the embeddings.

    Returns:
        Tuple[List[str], np.ndarray]:
            - A list of image IDs.
            - A NumPy array of shape (n_images, d) with dtype np.float32 containing all embeddings.
    """
    # Open and load the JSON file.
    with open(json_file, "r") as file:
        data = json.load(file)
    # Extract image IDs.
    image_ids = list(data.keys())
    # Convert each embedding list into a NumPy array with float32 data type.
    embeddings = [np.array(data[img_id], dtype=np.float32) for img_id in image_ids]
    # Stack embeddings into a single 2D NumPy array.
    embeddings_array = np.vstack(embeddings)
    # Ensure the array is stored in contiguous memory, which is required by FAISS.
    embeddings_array = np.ascontiguousarray(embeddings_array)
    return image_ids, embeddings_array


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Build a FAISS index using L2 (Euclidean) distance from the provided embeddings.

    Args:
        embeddings (np.ndarray): A 2D NumPy array of shape (n, d), where n is the number of embeddings
                                 and d is their dimensionality (should match the feature extractor output, e.g., 1000).

    Returns:
        faiss.Index: A FAISS index built using the IndexFlatL2 algorithm, with embeddings added.
    """
    # Get the dimensionality (d) of the embeddings.
    dimension = embeddings.shape[1]
    # Create a FAISS index that uses L2 distance.
    index = faiss.IndexFlatL2(dimension)
    # Add the embeddings to the index.
    index.add(embeddings)
    return index


def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int = 5) -> Tuple[List[float], List[int]]:
    """
    Search the FAISS index for the k nearest neighbors of the query embedding.

    Args:
        index (faiss.Index): The FAISS index built on the embeddings.
        query_embedding (np.ndarray): A 1D NumPy array representing the query embedding (of dimension d).
        k (int): The number of nearest neighbors to retrieve (default is 5).

    Returns:
        Tuple[List[float], List[int]]:
            - A list of distances to the k nearest neighbors.
            - A list of indices corresponding to the k nearest neighbors in the embeddings array.
    """
    # Expand dimensions of query_embedding to shape (1, d) for the search.
    query = np.expand_dims(query_embedding, axis=0)
    # Perform the search on the index.
    distances, indices = index.search(query, k)
    # Return the distances and indices as lists.
    return distances[0].tolist(), indices[0].tolist()


def main() -> None:
    """
    Main function to perform a FAISS search for similar images.

    The function performs the following steps:
      1. Loads image embeddings from "image_embeddings.json" and prints the number of images.
      2. Builds a FAISS index from the embeddings.
      3. Selects a query image from the "cleaned_images" folder.
      4. Processes the query image and extracts its embedding using the feature extraction model.
      5. Performs a search on the FAISS index to find the top k nearest neighbors.
      6. Prints the image IDs and distances for the top k similar images.
    """
    json_file: str = "image_embeddings.json"
    if not os.path.exists(json_file):
        print(f"Embeddings file {json_file} not found.")
        return

    # Load embeddings from JSON file.
    image_ids, embeddings = load_embeddings(json_file)
    print(f"Loaded embeddings for {len(image_ids)} images.")

    # Build a FAISS index using the loaded embeddings.
    index = build_faiss_index(embeddings)
    print("FAISS index built successfully.")

    # Specify the folder containing images.
    image_folder: str = "cleaned_images"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    # List all valid image files in the folder.
    image_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)])
    if not image_files:
        print(f"No valid images found in '{image_folder}'.")
        return

    # Select a query image (e.g., the first image in the sorted list).
    query_image_filename = image_files[0]
    query_image_path = os.path.join(image_folder, query_image_filename)
    print(f"Using query image: {query_image_filename}")

    # Process the query image to prepare it for the model.
    img_tensor = process_image(query_image_path)
    
    # Build the feature extraction model using FeatureExtractionCNN.
    model = FeatureExtractionCNN()
    model_path: str = "final_model/image_model.pt"
    # Load the saved model weights if available.
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded model weights for query image extraction.")
    else:
        print("Model weights not found; using default parameters.")
    model.eval()  # Set the model to evaluation mode.
    
    # Extract the embedding for the query image.
    with torch.no_grad():
        query_embedding = model(img_tensor).squeeze(0).numpy()
    
    # Ensure that the query embedding is contiguous and of type float32.
    if not query_embedding.flags['C_CONTIGUOUS']:
        query_embedding = np.ascontiguousarray(query_embedding)
    query_embedding = query_embedding.astype(np.float32)
    print(f"Query embedding shape: {query_embedding.shape}")

    # Search the FAISS index for the top k similar images.
    k: int = 5
    distances, indices = search_faiss(index, query_embedding, k)
    print(f"Top {k} similar images:")
    for dist, idx in zip(distances, indices):
        print(f"Image ID: {image_ids[idx]}, Distance: {dist:.4f}")


if __name__ == "__main__":
    main()
