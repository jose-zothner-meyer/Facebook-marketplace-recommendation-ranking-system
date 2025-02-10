"""
clean_images.py

This script processes images in the 'images/' folder by:
  1. Resizing images while maintaining aspect ratio.
  2. Adding padding to reach a target size.
  3. Converting images to RGB format.
  4. Saving the processed images to the 'cleaned_images/' folder.

Key differences from the initial version:
  - Uses a helper function to maintain aspect ratio and pad the image.
  - Default target size is set (adjustable via the final_size parameter).
"""

import os
from PIL import Image
from typing import Tuple

def resize_image(final_size: int, im: Image.Image) -> Image.Image:
    """
    Resize an image while maintaining aspect ratio and add padding to match the target size.
    
    Args:
        final_size (int): Target size for both width and height.
        im (Image.Image): The original image.
        
    Returns:
        Image.Image: The resized and padded image.
    """
    size = im.size  # Original (width, height)
    ratio = float(final_size) / max(size)  # Scale factor based on the largest dimension.
    new_image_size = tuple([int(x * ratio) for x in size])  # New size maintaining aspect ratio.

    im = im.resize(new_image_size, Image.LANCZOS)  # Resize with high-quality filter.
    new_im = Image.new("RGB", (final_size, final_size), (0, 0, 0))  # Create a black background.
    # Center the resized image on the new background.
    new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))

    return new_im

def clean_image_data(
    input_folder: str = "images",
    output_folder: str = "cleaned_images",
    final_size: int = 256  # Adjust this value as needed.
) -> None:
    """
    Process images by resizing (with padding) and converting to RGB, then save them to a new folder.
    
    Args:
        input_folder (str): Directory containing the original images.
        output_folder (str): Directory to save cleaned images.
        final_size (int): Target size for the images.
    """
    # Ensure the output folder exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each file in the input folder.
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Skip hidden files or non-image files.
        if filename.startswith('.') or not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        try:
            with Image.open(file_path) as img:
                # Convert image to RGB.
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize and pad the image.
                new_im = resize_image(final_size, img)

                # Save the processed image in the output folder.
                output_file_path = os.path.join(output_folder, filename)
                new_im.save(output_file_path)
                print(f"Saved cleaned image to: {output_file_path}")
        except Exception as e:
            print(f"Skipping file '{file_path}' due to error: {e}")

if __name__ == "__main__":
    # Run the cleaning process.
    clean_image_data()