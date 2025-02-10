"""
faiss_search.py

This script creates a FAISS search index using image embeddings and performs vector search to find similar images.
Key differences and integrations:
  - It loads image embeddings from 'data/output/image_embeddings.json' (produced by the integrated pipeline).
  - It loads the feature extraction model from 'data/final_model/image_model.pt'.
  - Inline comments explain each step.
"""

import os
import sys
import json
import faiss
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

# Function to extract features from a single image using the provided model and transform.
def extract_features(image_path, model, transform):
    try:
        image = Image.open(image_path).convert('RGB')
        # Add batch dimension.
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            features = model(image)
        return features.squeeze().numpy()
    except Exception as e:
        print(f"Error extracting features from image {image_path}: {e}")
        return None

# Step 1: Load image embeddings from the JSON file.
def load_image_embeddings(embeddings_file):
    try:
        with open(embeddings_file, 'r') as f:
            image_embeddings = json.load(f)
        return image_embeddings
    except Exception as e:
        print(f"Error loading image embeddings from {embeddings_file}: {e}")
        sys.exit(1)

# Step 2: Create a FAISS index and add the image embeddings.
def create_faiss_index(image_embeddings):
    try:
        # Determine embedding dimension from one of the embeddings.
        embedding_dim = len(next(iter(image_embeddings.values())))
        embeddings = np.array(list(image_embeddings.values())).astype('float32')
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)
        return index, list(image_embeddings.keys())
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        sys.exit(1)

# Step 3: Load the feature extraction model and define the necessary transformations.
def load_feature_extractor(model_path):
    try:
        weights = ResNet50_Weights.IMAGENET1K_V1
        feature_extractor_model = models.resnet50(weights=weights)
        num_features = feature_extractor_model.fc.in_features
        feature_extractor_model.fc = torch.nn.Linear(num_features, 1000)
        feature_extractor_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # Convert model to feature extractor by replacing fc with Identity.
        feature_extractor_model.fc = torch.nn.Identity()
        feature_extractor_model.eval()
        return feature_extractor_model
    except Exception as e:
        print(f"Error loading feature extractor model from {model_path}: {e}")
        sys.exit(1)

# Define the transformation pipeline.
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# Step 4: Perform vector search to find similar images.
def find_similar_images(image_path, model, transform, index, image_ids, k=5):
    try:
        query_embedding = extract_features(image_path, model, transform)
        if query_embedding is None:
            print("Error extracting features from query image.")
            return []
        distances, indices = index.search(np.array([query_embedding]), k)
        similar_image_ids = [image_ids[idx] for idx in indices[0]]
        return similar_image_ids
    except Exception as e:
        print(f"Error performing FAISS search: {e}")
        return []

def main():
    # Set query image path (for testing, you can modify this path).
    query_image_path = 'data/test_img/tv1.jpeg'
    if not os.path.exists(query_image_path):
        print(f"Error: Image file '{query_image_path}' does not exist.")
        sys.exit(1)

    # File paths for embeddings and model.
    embeddings_file = 'data/output/image_embeddings.json'
    model_path = 'data/final_model/image_model.pt'

    # Load components.
    image_embeddings = load_image_embeddings(embeddings_file)
    index, image_ids = create_faiss_index(image_embeddings)
    feature_extractor_model = load_feature_extractor(model_path)
    transform = get_transform()

    # Find similar images.
    similar_images = find_similar_images(
        query_image_path,
        feature_extractor_model,
        transform,
        index,
        image_ids,
        k=5
    )

    print(f"Similar images to '{query_image_path}': {similar_images}")

    # Clean up model from memory.
    del feature_extractor_model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
