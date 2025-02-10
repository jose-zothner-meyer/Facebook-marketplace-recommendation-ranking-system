"""
main.py

This script is the main entry point for the project. It performs the following tasks:
  1. Processes raw product and image data using ProductLabeler to generate a training CSV.
  2. Inspects the generated dataset (prints encoder/decoder mappings, a sample batch, total sample count, and a sample).
  3. Runs the full integrated transfer learning pipeline by calling run_pipeline() from pipeline.py.
  4. (Optional) You can monitor TensorBoard logs from the integrated pipeline.

Key differences from the initial version:
  - Instead of only using a legacy trainer, this file now calls run_pipeline() from pipeline.py,
    which executes the entire process (data processing, training with additional metrics, model conversion, and embedding extraction).
"""

import os
import pickle
from torch.utils.data import DataLoader

from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset
# Legacy trainer is still available if needed.
from a_resnet_transfer_trainer import ResNetTransferLearner
# Import the integrated pipeline function.
from pipeline import run_pipeline

CSV_PATH: str = "data/training_data.csv"

def ensure_training_data_exists(csv_path: str) -> None:
    """
    Verify that the training data CSV file exists.
    """
    if not os.path.exists(csv_path):
        print("❌ ERROR: training_data.csv was not found. Please run the data processing pipeline first.")
        exit(1)

def inspect_dataset() -> None:
    """
    Process product and image data to generate the training CSV, then load and inspect the dataset.
    
    Steps:
      1. Run ProductLabeler to create the CSV.
      2. Verify the CSV exists.
      3. Retrieve and print encoder/decoder mappings.
      4. Create an ImageDataset and DataLoader.
      5. Load and print one batch (image shape and labels).
      6. Print total sample count and a sample from index 111.
    """
    product_labeler = ProductLabeler(
        products_file="data/Cleaned_Products.csv",
        images_file="data/Images.csv",
        output_file=CSV_PATH
    )
    product_labeler.process()  # Process and generate training CSV
    ensure_training_data_exists(CSV_PATH)

    encoder = product_labeler.encoder
    decoder = product_labeler.decoder
    print("\nEncoder mapping:", encoder)
    print("Decoder mapping:", decoder)

    image_dir: str = "cleaned_images/"
    dataset = ImageDataset(CSV_PATH, image_dir)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    for images, labels in data_loader:
        print(f"\nBatch loaded: Image shape {images.shape}, Labels: {labels}")
        break

    print(f"\nTotal samples in dataset: {len(dataset)}")
    try:
        sample = dataset[111]
        print("Sample at index 111:", sample)
    except IndexError:
        print("Index 111 is out of range for the dataset.")

def main() -> None:
    """
    Main function to execute the complete project pipeline.
    
    It first inspects the dataset (using ProductLabeler) and then runs the integrated pipeline (via run_pipeline())
    which includes training (with additional metrics), model conversion, and embedding extraction.
    """
    print("=== Dataset Inspection ===")
    inspect_dataset()
    
    print("\n=== Running Full Transfer Learning Pipeline ===")
    run_pipeline()  # Run the integrated pipeline

if __name__ == "__main__":
    main()
