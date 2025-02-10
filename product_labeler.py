"""
product_labeler.py

This module defines the ProductLabeler class, which processes a cleaned products dataset,
assigns numeric labels to root categories, and integrates images data to generate a training CSV file.
It also creates encoder/decoder mappings for converting between category names and numeric labels.

Key differences:
  - Detailed inline comments explain each processing step.
  - The processing pipeline includes loading, cleaning, merging, and saving data.
"""

import pandas as pd
from typing import Dict, List

class ProductLabeler:
    """
    Processes a cleaned products dataset to assign numeric labels and integrate images data.
    
    Steps include:
      - Loading product and images CSV files.
      - Extracting the root category.
      - Creating encoder/decoder mappings.
      - Assigning numeric labels.
      - Merging images data.
      - Preparing and saving the final training dataset.
    """
    
    def __init__(self, products_file: str, images_file: str, output_file: str):
        """
        Initialize the ProductLabeler with file paths.
        
        Args:
            products_file (str): Path to the cleaned products CSV.
            images_file (str): Path to the images CSV.
            output_file (str): Path to save the processed training CSV.
        """
        self.products_file = products_file
        self.images_file = images_file
        self.output_file = output_file
        self.df_pdt = None       # DataFrame for products.
        self.df_images = None    # DataFrame for images.
        self.encoder = None      # Mapping: category string -> numeric label.
        self.decoder = None      # Mapping: numeric label -> category string.

    def load_data(self) -> None:
        """Load product and images CSV files."""
        print("Loading data...")
        self.df_pdt = pd.read_csv(self.products_file, lineterminator='\n')
        self.df_images = pd.read_csv(self.images_file, lineterminator='\n')
        print(f"Data loaded from {self.products_file} and {self.images_file}.")

    def extract_root_category(self) -> None:
        """
        Extract the root category from the 'category' column.
        The root category is obtained by splitting the category string at '/'.
        """
        print("Extracting root categories...")
        self.df_pdt['root_category'] = self.df_pdt['category'].apply(lambda cat: cat.split("/")[0].strip())
        print("Root categories extracted.")

    def create_encoder_decoder(self) -> None:
        """
        Create encoder and decoder mappings for root categories.
        """
        print("Creating encoder and decoder...")
        unique_categories: List[str] = self.df_pdt['root_category'].unique().tolist()
        self.encoder = {cat: idx for idx, cat in enumerate(unique_categories)}
        self.decoder = {idx: cat for cat, idx in self.encoder.items()}
        print("Encoder and decoder created.")

    def assign_labels(self) -> None:
        """
        Assign numeric labels to products based on the root category.
        """
        print("Assigning labels...")
        self.df_pdt['labels'] = self.df_pdt['root_category'].map(self.encoder)
        print("Labels assigned.")

    def merge_images(self) -> None:
        """
        Merge images data into the products DataFrame.
        
        The 'id' column in images is renamed to 'Image' and merged on product ID.
        """
        print("Merging images data...")
        self.df_images.rename(columns={'id': 'Image'}, inplace=True)
        self.df_pdt = self.df_pdt.merge(self.df_images, left_on='id', right_on='Image', how='left', suffixes=('', '_img'))
        print("Images data merged.")

    def prepare_training_data(self) -> None:
        """
        Prepare the final training dataset by selecting the image identifier and labels.
        """
        print("Preparing training data...")
        # Here, choose a column from images data as the image identifier.
        middle_column = self.df_images.columns[len(self.df_images.columns) // 2]
        self.df_pdt['Image'] = self.df_images[middle_column]
        self.df_pdt = self.df_pdt[['Image', 'labels']]
        print("Training data prepared.")

    def save_data(self) -> None:
        """Save the processed training DataFrame to CSV."""
        print(f"Saving labeled data to {self.output_file}...")
        self.df_pdt.to_csv(self.output_file, index=False)
        print(f"Labeled data saved to {self.output_file}.")

    def process(self) -> None:
        """
        Execute the full processing pipeline.
        """
        self.load_data()
        self.extract_root_category()
        self.create_encoder_decoder()
        self.assign_labels()
        self.merge_images()
        self.prepare_training_data()
        self.save_data()