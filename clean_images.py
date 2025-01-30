"""
clean_images.py

This script processes images in the 'images/' folder by:
1. Resizing all images while maintaining aspect ratio, padding to 512x512 pixels.
2. Converting all images to RGB format (ensuring consistency in channels).
3. Saving the cleaned images into a new folder named 'cleaned_images/'.

This ensures that all images are consistent in format and dimensions, making them
suitable for machine learning models and further processing.
"""

import os
from PIL import Image
from typing import Tuple

def resize_image(final_size: int, im: Image.Image) -> Image.Image:
    """
    Resizes an image while maintaining aspect ratio and adding padding to match the target size.

    Args:
        final_size (int): The target size for both width and height (e.g., 512 for 512x512).
        im (Image.Image): The original image.

    Returns:
        Image.Image: The resized and padded image.
    """
    size = im.size  # (width, height)
    ratio = float(final_size) / max(size)  # Scale factor to fit largest dimension
    new_image_size = tuple([int(x * ratio) for x in size])  # Maintain aspect ratio

    im = im.resize(new_image_size, Image.LANCZOS)  # Replaced ANTIALIAS with LANCZOS
    new_im = Image.new("RGB", (final_size, final_size), (0, 0, 0))  # Black background
    new_im.paste(im, ((final_size - new_image_size[0]) // 2, (final_size - new_image_size[1]) // 2))

    return new_im

def clean_image_data(
    input_folder: str = "images",
    output_folder: str = "cleaned_images",
    final_size: int = 256
) -> None:
    """
    Cleans an image dataset by resizing all images while maintaining aspect ratio and padding.

    Args:
        input_folder (str, optional): Path to the folder containing the original images. Defaults to 'images'.
        output_folder (str, optional): Path where cleaned images should be saved. Defaults to 'cleaned_images'.
        final_size (int, optional): The target size for images (width and height). Defaults to 512.

    Returns:
        None. The cleaned images are saved in the specified output folder.
    """
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each image in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)

        # Skip hidden/system files or non-image files
        if filename.startswith('.') or not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            continue

        try:
            # Open the image
            with Image.open(file_path) as img:
                # Convert to RGB format if not already
                if img.mode != "RGB":
                    img = img.convert("RGB")

                # Resize and pad the image
                new_im = resize_image(final_size, img)

                # Construct output file path (keep original file name)
                output_file_path = os.path.join(output_folder, filename)

                # Save cleaned image
                new_im.save(output_file_path)

                print(f"Saved cleaned image to: {output_file_path}")
        except Exception as e:
            print(f"Skipping file '{file_path}' due to error: {e}")

if __name__ == "__main__":
    # Run the cleaning process for all images in 'images/' and save to 'cleaned_images/'
    clean_image_data()
