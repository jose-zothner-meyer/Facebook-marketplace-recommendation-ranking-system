"""
feature_extractor_model.py

This module defines the FeatureExtractionCNN class and the transformation pipeline used for processing images.
It loads a pretrained ResNet50 model, removes its final classification layer, and adds a new fully connected layer 
to output a 1000-dimensional embedding.
"""

from typing import Any
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Define the transformation pipeline used during training.
TRANSFORM_PIPELINE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


class FeatureExtractionCNN(nn.Module):
    """
    FeatureExtractionCNN modifies a pretrained ResNet50 model to extract image embeddings.

    The model:
      - Loads a pretrained ResNet50.
      - Removes its final classification layer.
      - Adds a new fully connected layer mapping the 2048 features to a 1000-dimensional embedding.
    """
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_fc = nn.Linear(2048, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract image embeddings.

        Args:
            images (torch.Tensor): Batch of images with shape (batch_size, 3, 256, 256).

        Returns:
            torch.Tensor: Extracted embeddings with shape (batch_size, 1000).
        """
        x = self.features(images)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        return x
