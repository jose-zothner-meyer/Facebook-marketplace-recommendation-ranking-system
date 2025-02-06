"""
main.py

This script is the main entry point for the project. It performs the following tasks:
  1. Processes the raw product and image data using ProductLabeler to generate a training CSV.
  2. Inspects the generated dataset (prints encoder/decoder mappings, a sample batch, total sample count, and an example sample).
  3. Instantiates and runs the full transfer learning pipeline using ResNetTransferLearner.
  4. Loads and prints the decoder mapping (numeric label -> original category) for future inference.
"""

import os
import pickle
from torch.utils.data import DataLoader

from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset
from a_resnet_transfer_trainer import ResNetTransferLearner

CSV_PATH: str = "data/training_data.csv"


def ensure_training_data_exists(csv_path: str) -> None:
    """
    Verify that the training data CSV file exists.
    
    If the file is not found, print an error message and exit.
    """
    if not os.path.exists(csv_path):
        print("❌ ERROR: training_data.csv was not found. Please run the data processing pipeline first.")
        exit(1)


def inspect_dataset() -> None:
    """
    Process the product and image data to generate the training CSV file, then load and inspect the dataset.
    
    Steps performed:
      1. Run ProductLabeler to create the CSV file.
      2. Ensure that the CSV exists.
      3. Retrieve and print the encoder/decoder mappings.
      4. Create an ImageDataset and a DataLoader.
      5. Load a single batch and print its image tensor shape and labels.
      6. Print the total number of samples and a sample from index 111.
    """
    product_labeler = ProductLabeler(
        products_file="data/Cleaned_Products.csv",
        images_file="data/Images.csv",
        output_file=CSV_PATH
    )
    product_labeler.process()
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


def train_model() -> None:
    """
    Instantiate the transfer learning pipeline using ResNetTransferLearner and run the training.
    
    After training, load and print the decoder mapping (numeric label -> original category) for future inference.
    """
    trainer = ResNetTransferLearner()
    trainer.run()
    ensure_training_data_exists(CSV_PATH)
    decoder_path: str = "image_decoder.pkl"
    if os.path.exists(decoder_path):
        try:
            with open(decoder_path, "rb") as file:
                decoder = pickle.load(file)
            print("\nDecoder mapping (Numeric Label -> Category):")
            for key, value in decoder.items():
                print(f"{key} -> {value}")
        except Exception as error:
            print("Error loading decoder mapping:", error)
    else:
        print("Decoder mapping file not found.")


def main() -> None:
    """
    Main function to execute the data processing pipeline, inspect the dataset,
    and then train the model using transfer learning.
    """
    print("=== Dataset Inspection ===")
    inspect_dataset()
    
    print("\n=== Model Training ===")
    train_model()


if __name__ == "__main__":
    main()