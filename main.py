import os
from product_labeler import ProductLabeler
from image_dataset_pytorch import ImageDataset
from torch.utils.data import DataLoader

# Define the CSV path where the processed training data will be saved
csv_path = "data/training_data.csv"

def ensure_training_data_exists():
    """
    Verify that the training data CSV file exists.

    If the file is not found, an error message is printed and the program exits.
    """
    if not os.path.exists(csv_path):
        print("❌ ERROR: training_data.csv was not found. Ensure ProductLabeler is correctly processing data.")
        exit(1)

def main():
    """
    Main function to execute the data processing pipeline and load the image dataset.

    The following steps are performed:
      1. Process the product and images data to generate the training CSV file.
      2. Verify that the training data CSV file exists.
      3. Initialize the ImageDataset with the CSV and the image directory.
      4. Create a DataLoader to iterate over the dataset.
      5. Load a single batch and print its details.
      6. Print the total number of samples and an example sample from the dataset.
    """
    # Initialize ProductLabeler with the paths to the products, images, and output CSV files
    product_labeler = ProductLabeler(
        products_file="data/Cleaned_Products.csv",
        images_file="data/Images.csv",
        output_file=csv_path
    )
    
    # Run the complete data processing pipeline
    product_labeler.process()
    
    # Ensure that the processed training data file now exists
    ensure_training_data_exists()
    
    # (Optional) Retrieve encoder and decoder mappings if needed for later use
    encoder = product_labeler.encoder
    decoder = product_labeler.decoder

    # Specify the directory containing the cleaned images
    image_dir = "cleaned_images/"
    
    # Initialize the ImageDataset with the processed CSV file and image directory
    dataset = ImageDataset(csv_path, image_dir)
    
    # Create a DataLoader to load the dataset in batches (batch size of 1 for demonstration)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Load a single batch and print its image tensor shape and labels
    for images, labels in dataloader:
        print(f"Batch loaded: Image shape {images.shape}, Labels: {labels}")
        break

    # Print the total number of samples in the dataset
    print(f"Total samples in dataset: {len(dataset)}")
    
    # Print a sample from the dataset at index 111 (if available)
    try:
        sample = dataset[111]
        print("Sample at index 111:", sample)
    except IndexError:
        print("Index 111 is out of range for the dataset.")

if __name__ == "__main__":
    main()