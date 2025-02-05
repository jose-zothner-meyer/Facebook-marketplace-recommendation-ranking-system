import pandas as pd
import numpy as np
from PIL import Image
from typing import Dict, List

class ProductLabeler:
    """
    A class to process a cleaned products dataset, assign numeric labels
    to root categories, and integrate images data.

    The processing pipeline includes:
      1. Loading product and images data.
      2. Extracting the root category from the product's category field.
      3. Creating encoder/decoder mappings for the root categories.
      4. Assigning numeric labels to products.
      5. Merging images data with the product data.
      6. Preparing a training CSV file for further usage.
    """
    
    def __init__(self, products_file: str, images_file: str, output_file: str):
        """
        Initialize the ProductLabeler with paths for product data, images data, and the output file.

        Args:
            products_file (str): Path to the cleaned products CSV file.
            images_file (str): Path to the images CSV file.
            output_file (str): Path to save the processed training data CSV file.
        """
        # File paths for input and output
        self.products_file = products_file
        self.images_file = images_file
        self.output_file = output_file

        # DataFrames for storing product and images data (to be loaded later)
        self.df_pdt = None
        self.df_images = None

        # Encoder and decoder dictionaries for mapping root categories to numeric labels
        self.encoder = None
        self.decoder = None

    def load_data(self) -> None:
        """
        Load product and images data from CSV files into pandas DataFrames.
        """
        print("Loading data...")
        # Load products CSV file (specifying line terminator in case of non-standard newlines)
        self.df_pdt = pd.read_csv(self.products_file, lineterminator='\n')
        # Load images CSV file
        self.df_images = pd.read_csv(self.images_file, lineterminator='\n')
        print(f"Data loaded successfully from {self.products_file} and {self.images_file}.")

    def extract_root_category(self) -> None:
        """
        Extract the root category from the 'category' column in the products DataFrame.

        The root category is determined by splitting the category string at '/' and taking the first part.
        """
        print("Extracting root categories...")
        # Apply a lambda function to extract the first part of the category and remove extra spaces
        self.df_pdt['root_category'] = self.df_pdt['category'].apply(lambda category: category.split("/")[0].strip())
        print("Root categories extracted.")

    def create_encoder_decoder(self) -> None:
        """
        Create mappings for encoding and decoding root categories into numeric labels.

        The encoder maps each unique root category to a unique integer.
        The decoder is the inverse mapping from integer to category.
        """
        print("Creating encoder and decoder...")
        # Get a list of unique root categories
        unique_categories: List[str] = self.df_pdt['root_category'].unique().tolist()
        # Create encoder: maps category to a unique integer
        self.encoder: Dict[str, int] = {category: idx for idx, category in enumerate(unique_categories)}
        # Create decoder: maps integer back to its category string
        self.decoder: Dict[int, str] = {idx: category for category, idx in self.encoder.items()}
        print("Encoder and decoder created.")

    def assign_labels(self) -> None:
        """
        Assign numeric labels to each product based on its root category.

        The numeric label is obtained from the encoder and stored in a new 'labels' column.
        """
        print("Assigning labels...")
        # Map each root category to its corresponding numeric label using the encoder dictionary
        self.df_pdt['labels'] = self.df_pdt['root_category'].map(self.encoder)
        print("Labels assigned.")

    def merge_images(self) -> None:
        """
        Merge images data into the products DataFrame.

        The method renames the 'id' column in the images DataFrame to 'Image' for clarity,
        then merges it with the products DataFrame based on the product id.
        """
        print("Merging images data with product data...")
        # Rename the 'id' column in images DataFrame to 'Image'
        self.df_images.rename(columns={'id': 'Image'}, inplace=True)
        # Merge the images data with product data using the common identifier
        self.df_pdt = self.df_pdt.merge(
            self.df_images,
            left_on='id',
            right_on='Image',
            how='left',
            suffixes=('', '_img')
        )
        print("Images data merged successfully.")

    def prepare_training_data(self) -> None:
        """
        Prepare the final training dataset by selecting relevant columns.

        The method selects the image identifier (from the middle column of the images DataFrame)
        and the assigned labels, then reorders the DataFrame columns so that the image id is first.
        """
        print("Preparing training data...")
        # Select the middle column of the images DataFrame as the image identifier
        middle_column = self.df_images.columns[len(self.df_images.columns) // 2]
        # Assign the middle column to the 'Image' column in the products DataFrame
        self.df_pdt['Image'] = self.df_images[middle_column]
        # Reorder the columns so that 'Image' comes first followed by 'labels'
        self.df_pdt = self.df_pdt[['Image', 'labels']]
        print("Training data prepared.")

    def save_data(self) -> None:
        """
        Save the processed training data to the output CSV file.
        """
        print(f"Saving labeled data to {self.output_file}...")
        # Save the DataFrame to CSV without including the index
        self.df_pdt.to_csv(self.output_file, index=False)
        print(f"Labeled data saved to {self.output_file}.")

    def process(self) -> None:
        """
        Execute the complete processing pipeline:
          1. Load data from CSV files.
          2. Extract root category from product data.
          3. Create encoder and decoder mappings.
          4. Assign numeric labels to products.
          5. Merge images data with product data.
          6. Prepare the training data by selecting required columns.
          7. Save the processed data to an output CSV file.
        """
        self.load_data()
        self.extract_root_category()
        self.create_encoder_decoder()
        self.assign_labels()
        self.merge_images()
        self.prepare_training_data()
        self.save_data()

    def get_encoder_decoder(self) -> Dict[str, Dict]:
        """
        Retrieve the encoder and decoder mappings.

        Returns:
            Dict[str, Dict]: A dictionary containing:
                - 'encoder': Mapping from category string to numeric label.
                - 'decoder': Mapping from numeric label back to category string.
        """
        return {"encoder": self.encoder, "decoder": self.decoder}
