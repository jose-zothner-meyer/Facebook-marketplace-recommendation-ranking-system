"""
Image Embedding Extractor

This script loads a feature extraction model (based on a modified ResNet50),
processes a single image provided via the command line from the 'cleaned_images' folder,
extracts its embedding, and saves the embedding in a JSON file named 'image_embeddings.json'.

The JSON file maps the image id (derived from the image filename without extension)
to its embedding vector. This embedding can later be used to compare and find similar images.

Usage:
    python image_processor.py <image_filename>

Example:
    python image_processor.py my_image.jpg

The script applies the same transformations used during training (resizing, tensor conversion,
and normalization), and adds a batch dimension to the image tensor before passing it to the model.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

class FeatureExtractionCNN(nn.Module):
    """
    FeatureExtractionCNN modifies a pretrained ResNet50 model to extract image embeddings.

    This model performs the following operations:
    1. Loads a pretrained ResNet50 model.
    2. Removes the final classification (fully connected) layer.
    3. Adds a new fully connected layer that maps the 2048 features to a 1000-dimensional embedding.
    """
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        # Load a pretrained ResNet50 model.
        self.resnet50 = models.resnet50(pretrained=True)
        # Remove the final classification layer by taking all layers except the last one.
        self.resnet50 = nn.Sequential(*list(self.resnet50.children())[:-1])
        # Add a new fully connected layer to obtain a 1000-dimensional feature embedding.
        self.feature_fc = nn.Linear(2048, 1000) 

    def forward(self, imgs):
        """
        Performs a forward pass through the network to extract features.

        Args:
            imgs (torch.Tensor): A batch of images with shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: The extracted image embeddings with shape (batch_size, 1000).
        """
        # Pass the input images through the modified ResNet50 network.
        features = self.resnet50(imgs)  # Expected shape: (batch_size, 2048, 1, 1)
        # Flatten the features from shape (batch_size, 2048, 1, 1) to (batch_size, 2048).
        features = features.view(features.size(0), -1)
        # Pass the flattened features through the new fully connected layer.
        features = self.feature_fc(features)  # Resulting shape: (batch_size, 1000)
        return features

# Define the transformation pipeline.
# These transformations must match those applied during model training.
transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels.
    transforms.ToTensor(),          # Convert the PIL image to a PyTorch tensor (C, H, W).
    transforms.Normalize(           # Normalize using ImageNet's mean and standard deviation.
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def process_image(image_path):
    """
    Processes a single image by opening it, applying the necessary transformations,
    and adding a batch dimension.

    Args:
        image_path (str): The file path of the image.

    Returns:
        torch.Tensor or None: The processed image tensor of shape (1, 3, 256, 256),
                                or None if the image could not be opened.
    """
    try:
        # Open the image file and convert it to RGB mode.
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None
    # Apply the transformation pipeline to obtain a tensor of shape (3, 256, 256).
    img_tensor = transform_pipeline(image)
    # Add a batch dimension so that the tensor shape becomes (1, 3, 256, 256).
    return img_tensor.unsqueeze(0)

def main():
    """
    Main function to extract the embedding for a single image provided via the command line.
    
    The steps performed are:
    1. Validate that an image filename is provided as a command-line argument.
    2. Construct the path to the image within the 'cleaned_images' folder.
    3. Load the feature extraction model and load saved weights if available.
    4. Process the image using the defined transformation pipeline.
    5. Extract the image embedding using the feature extraction model.
    6. Save the resulting embedding in a JSON file mapping the image id to its embedding.
    """
    # Check that an image filename was provided.
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py cleaned_images/0a1baaa8-4556-4e07-a486-599c05cce76c.jpg")
        sys.exit(1)

    # Retrieve the image filename from the command-line arguments.
    image_filename = sys.argv[1]
    # Define the directory containing the cleaned images.
    image_folder = "cleaned_images"
    # Construct the full path to the image file.
    image_path = os.path.join(image_folder, image_filename)

    # Verify that the image file exists.
    if not os.path.exists(image_path):
        print(f"Image file not found: {image_path}")
        sys.exit(1)

    # Instantiate the feature extraction model.
    model = FeatureExtractionCNN()

    # Optionally, load saved model weights if the file exists.
    model_path = "final_model/image_model.pt"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        print("Loaded saved model weights.")
    else:
        print("Model weights not found. Using current model parameters.")

    # Set the model to evaluation mode to disable dropout and other training-specific layers.
    model.eval()

    # Process the image to get a properly transformed tensor.
    img_tensor = process_image(image_path)
    if img_tensor is None:
        sys.exit(1)

    # Use the model to extract the image embedding without computing gradients.
    with torch.no_grad():
        embedding = model(img_tensor)  # Expected shape: (1, 1000)

    # Remove the batch dimension so that the embedding has shape (1000,).
    embedding = embedding.squeeze(0)
    # Convert the PyTorch tensor to a list so that it can be saved in JSON format.
    embedding_list = embedding.tolist()

    # Derive an image id by removing the file extension from the image filename.
    image_id = os.path.splitext(image_filename)[0]
    # Create a dictionary mapping the image id to its embedding.
    image_embeddings = {image_id: embedding_list}

    # Save the dictionary of embeddings to a JSON file.
    output_json = "image_embeddings.json"
    with open(output_json, "w") as f:
        json.dump(image_embeddings, f)

    print(f"Saved embedding for image '{image_id}' to {output_json}")

if __name__ == "__main__":
    main()