import os
import sys
import torch
from torchvision import transforms
from PIL import Image

def process_image(image_path):
    """
    Process an image so it can be fed into the model.
    This includes:
      - Opening the image.
      - Applying the same transforms as used in training.
      - Adding a batch dimension to make its shape (1, C, H, W).
    """
    # Define the transformation pipeline (modify these if you used different transforms during training)
    transform_pipeline = transforms.Compose([
        # Resize the image to 256x256. (Or use CenterCrop if that was applied.)
        transforms.Resize((256, 256)),
        # Convert the PIL image to a tensor with shape (C, H, W) and pixel values in [0, 1]
        transforms.ToTensor(),
        # Normalize the image (example values, adjust if your training normalization is different)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Open the image and convert to RGB (in case it is in another mode)
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        sys.exit(1)
    
    # Apply the transformation pipeline to the image
    image_tensor = transform_pipeline(image)

    # At this point, image_tensor has shape (3, 256, 256).
    # Add a batch dimension to get shape (1, 3, 256, 256)
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor

def main():
    # Check if an image filename was provided as an argument
    if len(sys.argv) < 2:
        print("Usage: python image_processor.py <image_filename>")
        sys.exit(1)
    
    # Construct the full path to the image inside the cleaned_images folder
    image_filename = sys.argv[1]
    image_path = os.path.join("cleaned_images", image_filename)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)
    
    # Process the image
    processed_image = process_image(image_path)
    
    # For demonstration, print the shape to verify it matches (1, 3, 256, 256)
    print(f"Processed image shape: {processed_image.shape}")
    
    # Optionally, save or pass processed_image to your model
    # For example: output = model(processed_image)
    
if __name__ == "__main__":
    main()