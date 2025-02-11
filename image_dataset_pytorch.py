"""
image_dataset_pytorch.py

This module defines a custom PyTorch Dataset for loading images and their labels from a CSV file.
It applies specified transformations and returns the image tensor and label.
 
Key differences:
  - Includes data augmentation (random horizontal flip and rotation) and normalization using ImageNet statistics.
"""

import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    """
    Custom PyTorch Dataset for images and labels.
    """
    
    def __init__(self, csv_file: str, image_dir: str):
        """
        Initialize the dataset.
        
        Args:
            csv_file (str): Path to the CSV file containing image identifiers and labels.
            image_dir (str): Directory where images are stored.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # Data augmentation.
            transforms.RandomRotation(10),        # Data augmentation.
            transforms.ToTensor(),                # Convert image to tensor.
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize using ImageNet stats.
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self) -> int:
        """Return total number of samples."""
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Retrieve the image and label at a given index.
        
        Args:
            idx (int): Index of the sample.
            
        Returns:
            tuple: (image_tensor, label)
        """
        # Get image identifier (assumed to be in the first column) and construct file path.
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.image_dir, img_name + ".jpg")
        label = self.data.iloc[idx, 1]

        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        return image, label, img_name