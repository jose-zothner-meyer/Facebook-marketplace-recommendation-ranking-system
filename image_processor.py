import torch
import torchvision.transforms as transforms
from PIL import Image

def process_image(image_path, use_random_augmentations=False):
    """
    Load and preprocess the image, adding a batch dimension so the output shape is (1, channels, height, width).
    
    Args:
        image_path (str): The path to the image.
        use_random_augmentations (bool): If True, applies random augmentations (as used in training). 
                                         For inference, it's better to use deterministic transforms.
    
    Returns:
        torch.Tensor: A tensor of shape (1, 3, 224, 224) ready for model input.
    """
    # Open the image and convert to RGB.
    img = Image.open(image_path).convert("RGB")
    
    if use_random_augmentations:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # For inference, deterministic transforms are usually preferable.
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    # Apply the transformation pipeline.
    img_tensor = transform(img)
    
    # Add a batch dimension: shape becomes (1, channels, height, width).
    input_tensor = img_tensor.unsqueeze(0)
    return input_tensor

# Example usage:
if __name__ == "__main__":
    image_path = "cleaned_images/"  # Replace with your actual image path.
    processed_tensor = process_image(image_path, use_random_augmentations=False)
    print("Processed tensor shape:", processed_tensor.shape)
