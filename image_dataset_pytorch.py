"""
image_dataset_pytorch.py

This module defines the custom ImageDataset class for PyTorch.
It loads image file names and corresponding labels from a CSV file,
applies necessary transformations, and returns the image tensor and label.
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    """
    A custom PyTorch Dataset for loading and transforming images along with their labels.
    """
    
    def __init__(self, csv_file: str, image_dir: str):
        """
        Initialize the dataset.
        
        Args:
            csv_file (str): Path to the CSV file containing image identifiers and labels.
            image_dir (str): Directory where the image files are stored.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Data augmentation: random horizontal flip
            transforms.RandomRotation(10),        # Data augmentation: random rotation
            transforms.ToTensor(),                # Convert image to PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """Return the total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve the image and label at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            tuple: (image_tensor, label)
        """
        # Get the image identifier (assumed to be the first column) and construct the image path
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        label = self.data.iloc[idx, 1]

        # Open the image and convert to RGB
        image = Image.open(img_path).convert("RGB")
        
        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)

        return image, label
