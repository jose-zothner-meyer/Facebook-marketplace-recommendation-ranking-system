"""
main.py

A top-level script to run all data processing tasks:
1. Data cleaning (via the ProdClean class from clean_tabular_data.py).
2. Product labeling (via the ProductLabeler class from product_labeler.py).
3. Image dataset preparation (via ImageDataset for PyTorch).
"""

from clean_tabular_data import ProdClean
from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset  # Import your dataset class
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

def main() -> None:
    """
    Main entry point for the data processing pipeline.

    Steps:
    ------
    1) Clean the products data using ProdClean.
    2) Label the cleaned data using ProductLabeler.
    3) Load the labeled dataset into a PyTorch dataset (ImageDataset).
    4) Create a DataLoader for batch processing.
    """

    # 1. Clean the data
    cleaner = ProdClean()
    input_file = "data/Products.csv"  # Raw product file
    cleaned_file = "data/Cleaned_Products.csv"

    try:
        print("Starting data cleaning process...")
        cleaned_df = cleaner.data_clean(input_file=input_file, line_terminator="\n")

        cleaned_df.to_csv(cleaned_file, index=False)
        print(f"Data cleaning complete! Cleaned file saved to: {cleaned_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        return

    # 2. Label the cleaned data
    labeled_file = "data/training_data.csv"
    images_file = "data/Images.csv"  # Ensure you have a file linking images to products

    try:
        print("\nStarting labeling process...")
        labeler = ProductLabeler(
            products_file=cleaned_file,
            images_file=images_file,  # Ensure image linking is processed
            output_file=labeled_file
        )
        labeler.process()  # Runs the full labeling pipeline

        # Print encoder/decoder
        mappings = labeler.get_encoder_decoder()
        encoder = mappings["encoder"]
        decoder = mappings["decoder"]

        print("\nLabeling complete!")
        print("Encoder (Category -> Integer):", encoder)
        print("Decoder (Integer -> Category):", decoder)

    except FileNotFoundError:
        print(f"Error: The file '{cleaned_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred during labeling: {e}")
        return

    # 3. Load Image Dataset for PyTorch
    image_dir = "data/images/"  # Path to folder where images are stored

    # Define transformations for the dataset
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # Data augmentation
        transforms.ToTensor(),  # Convert image to PyTorch tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    print("\nLoading image dataset for PyTorch...")
    dataset = ImageDataset(csv_file=labeled_file, image_dir=image_dir, transform=transform)

    # 4. Create a DataLoader for batch processing
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Verify DataLoader is working correctly
    print(f"Dataset size: {len(dataset)}")

    for images, labels in dataloader:
        print(f"Batch loaded: Image shape {images.shape}, Labels: {labels}")
        break

    print("\nImage dataset successfully loaded into PyTorch!")

if __name__ == "__main__":
    main()
