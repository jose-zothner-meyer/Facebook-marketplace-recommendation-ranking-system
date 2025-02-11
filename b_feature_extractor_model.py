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