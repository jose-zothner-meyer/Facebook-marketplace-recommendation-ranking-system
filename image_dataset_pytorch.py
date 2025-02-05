import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# Define the CSV path containing the training data (image names and labels)
csv_path = "data/training_data.csv"

class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset class for loading, transforming, and returning images with labels.

    This dataset:
      - Loads image names and labels from a CSV file.
      - Applies a sequence of transformations (data augmentation and normalization) to the images.
      - Returns image tensors along with their corresponding labels.
    """
    
    def __init__(self, csv_file: str, image_dir: str):
        """
        Initialize the ImageDataset.

        Args:
            csv_file (str): Path to the CSV file containing image names and labels.
            image_dir (str): Directory where image files are stored.
        """
        # Load the CSV data into a pandas DataFrame
        self.data = pd.read_csv(csv_file)
        # Save the image directory path
        self.image_dir = image_dir
        
        # Define the image transformation pipeline
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Random horizontal flip for augmentation
            transforms.RandomRotation(10),        # Random rotation (up to 10 degrees)
            transforms.ToTensor(),                # Convert image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet statistics
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """
        Return the total number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve an image and its corresponding label given an index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple (image, label) where:
                - image: A transformed image tensor.
                - label: The corresponding label as an integer.
        """
        # Get the image name from the first column of the CSV
        img_name = self.data.iloc[idx, 0]
        # Construct the full path to the image file (assuming '.jpg' extension)
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        # Retrieve the label from the second column of the CSV
        label = self.data.iloc[idx, 1]
        
        # Open the image and convert it to RGB format
        image = Image.open(img_path).convert("RGB")
        
        # Apply the transformation pipeline if defined
        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == "__main__":
    # When running this file directly, test the dataset functionality.
    IMAGE_DIR = "cleaned_images/"  # Directory containing the images
    dataset = ImageDataset(csv_path, IMAGE_DIR)
    
    # Create a DataLoader to load data in batches for testing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Retrieve and print the shape and labels of one batch
    for images, labels in dataloader:
        print("Batch shape:", images.shape, "Labels:", labels)
        break
