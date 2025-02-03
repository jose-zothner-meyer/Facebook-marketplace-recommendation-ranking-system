import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from product_labeler import ProductLabeler

# Load the CSV file containing image paths and labels
csv_path = "data/training_data.csv"

def ensure_training_data_exists():
    if not os.path.exists(csv_path):
        print("❌ ERROR: training_data.csv was not found. Ensure ProductLabeler is correctly processing data.")
        exit(1)

# Initialize ProductLabeler and process data
product_labeler = ProductLabeler(products_file="data/Cleaned_Products.csv", images_file="data/Images.csv", output_file=csv_path)
product_labeler.load_data()  # Ensure data is loaded before encoding
product_labeler.extract_root_category()
product_labeler.create_encoder_decoder()
product_labeler.assign_labels()
product_labeler.process()

# Ensure training_data.csv was created before proceeding
ensure_training_data_exists()

# Retrieve encoder and decoder
encoder = product_labeler.encoder  # Use encoder from ProductLabeler
decoder = product_labeler.decoder  # Use decoder from ProductLabeler

# Define transformations for data augmentation and normalization
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally for augmentation
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize pixel values
])

class ImageDataset(Dataset):
    """Custom PyTorch Dataset class to handle image loading and processing."""
    
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Initializes the dataset by loading image file names and labels from a CSV file.
        
        Args:
            csv_file (str): Path to the CSV file with image names and labels.
            image_dir (str): Directory where images are stored.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)  # Load dataset from CSV
        self.image_dir = image_dir  # Store image directory path
        self.transform = transform  # Store transformation function

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieves an image and its corresponding label based on the index.
        
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            tuple: (image, label), where image is a transformed tensor and label is an integer.
        """
        img_name = self.data.iloc[idx, 0]  # Get image file name from CSV
        img_path = os.path.join(self.image_dir, img_name + ".jpg")  # Construct full image path
        label = self.data.iloc[idx, 1]  # Retrieve category label
        label_encoded = encoder.get(label, -1)  # Convert category label to numerical value, default to -1 if not found
        
        # Load image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if any are provided
        if self.transform:
            image = self.transform(image)

        return image, label_encoded

# Initialize dataset with the given image directory
image_dir = "cleaned_images/"  # Change this to the actual path where images are stored
dataset = ImageDataset(csv_path, image_dir, transform=transform)

# Create a DataLoader to load data in batches
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)  # Shuffle to improve training

# Example usage: Load a single batch and print its shape and labels
for images, labels in dataloader:
    print(f"Batch loaded: Image shape {images.shape}, Labels: {labels}")
    break