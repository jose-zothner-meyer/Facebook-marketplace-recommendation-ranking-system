"""
FAISS Image Search

This script loads a dictionary of image embeddings from a JSON file,
builds a FAISS index for vector search, and demonstrates how to search for
images similar to a query image.

Requirements:
- The JSON file "image_embeddings.json" must exist and contain a dictionary
  mapping image IDs (filenames without extension) to their corresponding
  embedding vectors.
- A feature extraction model (based on a modified ResNet50) must be defined
  to extract embeddings from any query image.
- The saved model weights for feature extraction are expected at:
  "final_model/image_model.pt".
- The images are assumed to reside in the "cleaned_images" folder.

Usage:
    python faiss_search.py

The script prints the top-k similar image IDs along with their distances.
"""

import os
import json
import numpy as np
import faiss  # Make sure you have installed faiss via: pip install faiss-cpu (or faiss-gpu)
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

###############################################################################
# Feature Extraction Model Definition
###############################################################################
class FeatureExtractionCNN(nn.Module):
    """
    FeatureExtractionCNN modifies a pretrained ResNet50 model to extract image embeddings.
    
    The model:
    1. Loads a pretrained ResNet50.
    2. Removes the final classification layer.
    3. Adds a new fully connected layer mapping the 2048 features to a 1000-dimensional embedding.
    """
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        # Load the pretrained ResNet50 model.
        self.resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer.
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        # Add a new fully connected layer to get a 1000-dimensional embedding.
        self.feature_fc = nn.Linear(2048, 1000)

    def forward(self, imgs):
        """
        Forward pass to extract embeddings.
        
        Args:
            imgs (torch.Tensor): A batch of images of shape (batch_size, 3, 256, 256).
        
        Returns:
            torch.Tensor: The extracted embeddings with shape (batch_size, 1000).
        """
        features = self.resnet50(imgs)  # Expected shape: (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (batch_size, 2048)
        features = self.feature_fc(features)  # Map to (batch_size, 1000)
        return features

###############################################################################
# Image Preprocessing Function
###############################################################################
# Define the transformation pipeline (must match the one used during training).
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize image to 256x256.
    transforms.ToTensor(),          # Convert image to tensor with shape (C, H, W).
    transforms.Normalize(           # Normalize using ImageNet statistics.
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image(image_path):
    """
    Processes an image: loads it, applies transformations, and adds a batch dimension.
    
    Args:
        image_path (str): Path to the image file.
    
    Returns:
        torch.Tensor or None: Processed image tensor of shape (1, 3, 256, 256) or None on error.
    """
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    img_tensor = transform_pipeline(image)
    return img_tensor.unsqueeze(0)  # Add batch dimension

###############################################################################
# FAISS Index Building and Search Functions
###############################################################################
def load_embeddings(json_file):
    """
    Loads the image embeddings dictionary from a JSON file.
    
    Args:
        json_file (str): Path to the JSON file containing image embeddings.
    
    Returns:
        tuple: (image_ids, embeddings) where image_ids is a list of image IDs and 
               embeddings is a NumPy array of shape (n_images, embedding_dim).
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    image_ids = list(data.keys())
    # Convert each embedding list into a NumPy array of type float32.
    embeddings = [np.array(data[img_id], dtype=np.float32) for img_id in image_ids]
    embeddings = np.vstack(embeddings)  # Shape: (n_images, embedding_dim)
    return image_ids, embeddings

def build_faiss_index(embeddings):
    """
    Builds a FAISS index using L2 distance from the given embeddings.
    
    Args:
        embeddings (np.ndarray): NumPy array of shape (n, d) where d is the embedding dimension.
    
    Returns:
        faiss.Index: The FAISS index with the embeddings added.
    """
    d = embeddings.shape[1]  # Embedding dimension (should be 1000)
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)  # Add all embeddings to the index.
    return index

def search_faiss(index, query_embedding, k=5):
    """
    Searches the FAISS index for the k nearest neighbors of the query embedding.
    
    Args:
        index (faiss.Index): The FAISS index.
        query_embedding (np.ndarray): Query embedding of shape (d,).
        k (int): Number of nearest neighbors to return.
    
    Returns:
        tuple: (distances, indices) where distances is a list of distances and indices is a list
               of integer indices corresponding to the matched image embeddings.
    """
    query_embedding = np.expand_dims(query_embedding, axis=0)  # Shape: (1, d)
    distances, indices = index.search(query_embedding, k)
    return distances[0], indices[0]

###############################################################################
# Main Function: Load Embeddings, Build FAISS Index, and Perform Search
###############################################################################
def main():
    """
    Main function to:
    1. Load image embeddings from a JSON file.
    2. Build a FAISS index using the embeddings.
    3. Process a query image and extract its embedding.
    4. Use FAISS to search for similar images.
    5. Print the top-k similar image IDs and their distances.
    """
    # --- Step 1: Load the Saved Image Embeddings ---
    json_file = "image_embeddings.json"
    if not os.path.exists(json_file):
        print(f"Embeddings file {json_file} not found.")
        return
    image_ids, embeddings = load_embeddings(json_file)
    print(f"Loaded embeddings for {len(image_ids)} images.")

    # --- Step 2: Build the FAISS Index ---
    index = build_faiss_index(embeddings)
    print("FAISS index built successfully.")

    # --- Step 3: Process a Query Image ---
    # For demonstration, we use the first image in the cleaned_images folder as the query.
    image_folder = "cleaned_images"
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]
    if not files:
        print(f"No valid image files found in '{image_folder}'.")
        return
    # Select the first image (alphabetically)
    query_image_filename = sorted(files)[0]
    query_image_path = os.path.join(image_folder, query_image_filename)
    print(f"Using query image: {query_image_filename}")

    img_tensor = process_image(query_image_path)
    if img_tensor is None:
        print("Failed to process the query image.")
        return

    # --- Step 4: Extract the Query Image Embedding ---
    model = FeatureExtractionCNN()
    # Load saved model weights if available.
    model_path = "final_model/image_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded model weights for query image extraction.")
    else:
        print("Model weights not found; using current model parameters.")
    model.eval()
    with torch.no_grad():
        query_embedding = model(img_tensor).squeeze(0).numpy()  # Shape: (1000,)

    # --- Step 5: Perform FAISS Search ---
    k = 5  # Number of similar images to retrieve.
    distances, indices = search_faiss(index, query_embedding, k)
    print(f"Top {k} similar images:")
    for dist, idx in zip(distances, indices):
        print(f"Image ID: {image_ids[idx]}, Distance: {dist}")

if __name__ == "__main__":
    main()