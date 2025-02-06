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
from resnet_transfer_trainer import ResNetTransferLearner

# Define the CSV path where the processed training data will be saved.
csv_path = "data/training_data.csv"

def ensure_training_data_exists(csv_path):
    """
    Verify that the training data CSV file exists.
    
    If the file is not found, print an error message and exit.
    """
    if not os.path.exists(csv_path):
        print("❌ ERROR: training_data.csv was not found. Please run the data processing pipeline first.")
        exit(1)

def inspect_dataset():
    """
    Process the product and image data to generate the training CSV file, then load and inspect the dataset.
    
    Steps performed:
      1. Initialize and run ProductLabeler to create the CSV file.
      2. Ensure that the CSV exists.
      3. Retrieve and print the encoder/decoder mappings.
      4. Create an ImageDataset and a DataLoader.
      5. Load a single batch and print its image tensor shape and labels.
      6. Print the total number of samples and a sample from index 111.
    """
    # Initialize ProductLabeler with the paths to the products, images, and output CSV.
    product_labeler = ProductLabeler(
        products_file="data/Cleaned_Products.csv",
        images_file="data/Images.csv",
        output_file=csv_path
    )
    
    # Run the complete data processing pipeline.
    product_labeler.process()
    
    # Ensure that the processed training data file exists.
    ensure_training_data_exists(csv_path)
    
    # Retrieve and print encoder and decoder mappings.
    encoder = product_labeler.encoder
    decoder = product_labeler.decoder
    print("\nEncoder mapping:", encoder)
    print("Decoder mapping:", decoder)
    
    # Specify the directory containing the cleaned images.
    image_dir = "cleaned_images/"
    
    # Initialize the ImageDataset with the processed CSV file and image directory.
    dataset = ImageDataset(csv_path, image_dir)
    
    # Create a DataLoader to load the dataset (using a batch size of 1 for demonstration).
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Load a single batch and print its image tensor shape and labels.
    for images, labels in dataloader:
        print(f"\nBatch loaded: Image shape {images.shape}, Labels: {labels}")
        break

    # Print the total number of samples in the dataset.
    print(f"\nTotal samples in dataset: {len(dataset)}")
    
    # Print a sample from the dataset at index 111 (if available).
    try:
        sample = dataset[111]
        print("Sample at index 111:", sample)
    except IndexError:
        print("Index 111 is out of range for the dataset.")

def train_model():
    """
    Instantiate the transfer learning pipeline using ResNetTransferLearner and run the training.
    
    After training, load and print the decoder mapping (numeric label -> original category) for future inference.
    """
    # Instantiate the transfer learner using default hyperparameters.
    trainer = ResNetTransferLearner()
    
    # Run the full training pipeline:
    #   - Process data, set up DataLoaders, configure the model (with the last two layers unfrozen and wrapped),
    #     and train the model while saving weights and metrics.
    trainer.run()
    
    # Ensure that the training CSV exists after processing.
    ensure_training_data_exists(csv_path)
    
    # Load and print the decoder mapping (saved as "image_decoder.pkl").
    decoder_path = "image_decoder.pkl"
    if os.path.exists(decoder_path):
        try:
            with open(decoder_path, "rb") as f:
                decoder = pickle.load(f)
            print("\nDecoder mapping (Numeric Label -> Category):")
            for key, value in decoder.items():
                print(f"{key} -> {value}")
        except Exception as e:
            print("Error loading decoder mapping:", e)
    else:
        print("Decoder mapping file not found.")

def main():
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