"""
b_feature_extractor_model.py

This module defines the FeatureExtractionCNN class and the transformation pipeline used for processing images.
Key differences from your initial code:
  - Inline comments clarify that the final classification layer is removed so that the network outputs a 1000-dimensional embedding.
"""

from typing import Any
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# Transformation pipeline used during feature extraction.
TRANSFORM_PIPELINE = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class FeatureExtractionCNN(nn.Module):
    """
    Modifies a pretrained ResNet50 to extract image embeddings.
    
    The final classification layer is removed so that a 1000-dimensional embedding is produced.
    """
    def __init__(self) -> None:
        super(FeatureExtractionCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer by taking all layers except the last one.
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        # Add a new fully connected layer to map features to a 1000-dimensional embedding.
        self.feature_fc = nn.Linear(2048, 1000)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: extract features and compute a 1000-dimensional embedding.
        """
        x = self.features(images)
        x = x.view(x.size(0), -1)
        x = self.feature_fc(x)
        return x
