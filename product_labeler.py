"""
product_labeler.py

This module defines the ProductLabeler class, which processes a cleaned products dataset,
assigns numeric labels to root categories, and integrates images data to generate a training CSV file.
It also creates encoder/decoder mappings for converting between category names and numeric labels.
"""

import pandas as pd
from typing import Dict, List

class ProductLabeler:
    """
    A class to process a cleaned products dataset, assign numeric labels to root categories,
    and integrate images data. The processing pipeline includes:
      1. Loading product and images data.
      2. Extracting the root category.
      3. Creating encoder/decoder mappings.
      4. Assigning numeric labels.
      5. Merging images data.
      6. Preparing the final training dataset.
      7. Saving the processed data.
    """
    
    def __init__(self, products_file: str, images_file: str, output_file: str):
        """
        Initialize the ProductLabeler with paths for the product data, images data, and output CSV.
        
        Args:
            products_file (str): Path to the cleaned products CSV file.
            images_file (str): Path to the images CSV file.
            output_file (str): Path to save the processed training data CSV file.
        """
        self.products_file = products_file
        self.images_file = images_file
        self.output_file = output_file
        self.df_pdt = None       # DataFrame for product data
        self.df_images = None    # DataFrame for images data
        self.encoder = None      # Mapping from category string to numeric label
        self.decoder = None      # Mapping from numeric label back to category string

    def load_data(self) -> None:
        """Load the product and images CSV files into DataFrames."""
        print("Loading data...")
        self.df_pdt = pd.read_csv(self.products_file, lineterminator='\n')
        self.df_images = pd.read_csv(self.images_file, lineterminator='\n')
        print(f"Data loaded successfully from {self.products_file} and {self.images_file}.")

    def extract_root_category(self) -> None:
        """
        Extract the root category from the 'category' column in the products DataFrame.
        
        The root category is obtained by splitting the string at '/' and taking the first part.
        """
        print("Extracting root categories...")
        self.df_pdt['root_category'] = self.df_pdt['category'].apply(lambda cat: cat.split("/")[0].strip())
        print("Root categories extracted.")

    def create_encoder_decoder(self) -> None:
        """
        Create encoder and decoder mappings for the root categories.
        
        The encoder maps each unique category to a unique integer.
        The decoder is the inverse mapping.
        """
        print("Creating encoder and decoder...")
        unique_categories: List[str] = self.df_pdt['root_category'].unique().tolist()
        self.encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.decoder = {idx: cat for cat, idx in self.encoder.items()}
        print("Encoder and decoder created.")

    def assign_labels(self) -> None:
        """
        Assign numeric labels to the products using the encoder mapping.
        
        A new column 'labels' is added to the product DataFrame.
        """
        print("Assigning labels...")
        self.df_pdt['labels'] = self.df_pdt['root_category'].map(self.encoder)
        print("Labels assigned.")

    def merge_images(self) -> None:
        """
        Merge the images data into the products DataFrame.
        
        The 'id' column in the images DataFrame is renamed to 'Image' to avoid confusion,
        and then the data is merged on the product ID.
        """
        print("Merging images data with product data...")
        self.df_images.rename(columns={'id': 'Image'}, inplace=True)
        self.df_pdt = self.df_pdt.merge(self.df_images, left_on='id', right_on='Image', how='left', suffixes=('', '_img'))
        print("Images data merged successfully.")

    def prepare_training_data(self) -> None:
        """
        Prepare the final training dataset by selecting the image identifier and labels.
        
        The 'Image' column is set using one of the columns from the images DataFrame,
        and the DataFrame is reordered to have the image identifier first.
        """
        print("Preparing training data...")
        middle_column = self.df_images.columns[len(self.df_images.columns) // 2]
        self.df_pdt['Image'] = self.df_images[middle_column]
        self.df_pdt = self.df_pdt[['Image', 'labels']]
        print("Training data prepared.")

    def save_data(self) -> None:
        """Save the processed training DataFrame to the output CSV file."""
        print(f"Saving labeled data to {self.output_file}...")
        self.df_pdt.to_csv(self.output_file, index=False)
        print(f"Labeled data saved to {self.output_file}.")

    def process(self) -> None:
        """
        Execute the full processing pipeline:
          1. Load data.
          2. Extract root category.
          3. Create encoder and decoder.
          4. Assign labels.
          5. Merge images data.
          6. Prepare training data.
          7. Save the processed data.
        """
        self.load_data()
        self.extract_root_category()
        self.create_encoder_decoder()
        self.assign_labels()
        self.merge_images()
        self.prepare_training_data()
        self.save_data()
