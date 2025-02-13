import json
import numpy as np
import faiss
from PIL import Image
import torch
import torchvision.transforms as transforms

# Import the model to extract features
from a_resnet_transfer_trainer import FineTunedResNet
import json

# Load the saved dictionary containing {image_id: embedding}
with open('data/output/image_embeddings.json', 'r') as f:
    embeddings_dict = json.load(f)

image_ids = list(embeddings_dict.keys())
embeddings = np.array(list(embeddings_dict.values()), dtype=np.float32)

# Create a FAISS index
dimension = embeddings.shape[1]  # Dimension of each embedding vector
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Example usage: query the index with the first embedding (for testing)
D, I = index.search(embeddings[:1], k=5)
similar_images = [image_ids[idx] for idx in I[0]]
print("Similar images (from stored embeddings) to the first image:", similar_images)


# ---------------------------------------------------------------------
# Procedure to load an image from a given path and perform FAISS search
def extract_embedding(image_path):
    """
    Loads an image from `image_path`, applies the necessary transforms, extracts a feature
    embedding using the pre-trained model, and returns the embedding as a numpy array.
    """
    # Define transforms (should match the ones used during training)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    # Open and transform the image
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension

    # Set device: use GPU if available else CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model and convert to a feature extractor (classification head removed)
    num_classes = 13  # update if different
    saved_weights = 'data/model_evaluation/model_20250211-192554/weights/epoch_5.pth'  # update if needed
    
    model_training = FineTunedResNet(num_classes)
    model_training.load_state_dict(torch.load(saved_weights, map_location=device))
    model_training.to(device)
    
    # Remove the classification head; adjust according to your model structure
    model_extractor = torch.nn.Sequential(*list(model_training.combined_model.children())[:-1])
    model_extractor.to(device)
    model_extractor.eval()
    
    with torch.no_grad():
        embedding = model_extractor(image_tensor.to(device))
        embedding = embedding.flatten().cpu().numpy()
    
    return embedding


# Provide the path to your query image
query_image_path = 'images/0a40eb4e-b033-4a98-9374-e1d4ad2943ac.jpg'  # Update this path to your image

# Extract the embedding for the query image
query_embedding = extract_embedding(query_image_path)
query_embedding = query_embedding.reshape(1, dimension)

# Search the FAISS index using the extracted query embedding
D, I = index.search(query_embedding, k=5)
similar_images = [image_ids[idx] for idx in I[0]]
print("Similar images for the query image:", similar_images)