import pandas as pd
from typing import Dict, List

class ProductLabeler:
    """
    A class to process a cleaned products dataset, assign numeric labels
    to root categories, and integrate images data.
    """
    def __init__(self, products_file: str, images_file: str, output_file: str):
        """
        Initialize the ProductLabeler with input and output file paths.

        Args:
            products_file (str): Path to the cleaned products CSV file.
            images_file (str): Path to the images CSV file.
            output_file (str): Path to save the processed file with labels and images.
        """
        self.products_file = products_file
        self.images_file = images_file
        self.output_file = output_file
        self.df_pdt = None
        self.df_images = None
        self.encoder = None
        self.decoder = None

    def load_data(self) -> None:
        """Load the cleaned products and images data into DataFrames."""
        print("Loading data...")
        self.df_pdt = pd.read_csv(self.products_file, lineterminator='\n')
        self.df_images = pd.read_csv(self.images_file, lineterminator='\n')
        print(f"Data loaded successfully from {self.products_file} and {self.images_file}.")

    def extract_root_category(self) -> None:
        """Extract the root category from the 'category' column."""
        print("Extracting root categories...")
        self.df_pdt['root_category'] = self.df_pdt['category'].str.split('/').str[0].str.strip()
        print("Root categories extracted.")

    def create_encoder_decoder(self) -> None:
        """Create an encoder and decoder for root categories."""
        print("Creating encoder and decoder...")
        unique_categories: List[str] = self.df_pdt['root_category'].unique().tolist()
        self.encoder: Dict[str, int] = {category: idx for idx, category in enumerate(unique_categories)}
        self.decoder: Dict[int, str] = {idx: category for category, idx in self.encoder.items()}
        print("Encoder and decoder created.")

    def assign_labels(self) -> None:
        """Assign numeric labels to root categories."""
        print("Assigning labels...")
        self.df_pdt['labels'] = self.df_pdt['root_category'].map(self.encoder)
        print("Labels assigned.")

    def merge_images(self) -> None:
        """Merge the images dataset with the product dataset."""
        print("Merging images data with product data...")
        self.df_pdt.rename(columns={'id': 'product_id'}, inplace=True)
        self.df_pdt = self.df_pdt.merge(self.df_images, on='product_id', how='left')
        print("Images data merged successfully.")

    def save_data(self) -> None:
        """Save the updated DataFrame with labels and images to the output file."""
        print(f"Saving labeled data to {self.output_file}...")
        self.df_pdt.to_csv(self.output_file, index=False)
        print(f"Labeled data saved to {self.output_file}.")

    def process(self) -> None:
        """
        Execute the entire processing pipeline:
        1. Load data.
        2. Extract root category.
        3. Create encoder and decoder.
        4. Assign labels.
        5. Merge images data.
        6. Save the updated data.
        """
        self.load_data()
        self.extract_root_category()
        self.create_encoder_decoder()
        self.assign_labels()
        self.merge_images()
        self.save_data()

    def get_encoder_decoder(self) -> Dict[str, Dict]:
        """
        Retrieve the encoder and decoder dictionaries.

        Returns:
            Dict[str, Dict]: A dictionary containing the encoder and decoder.
        """
        return {"encoder": self.encoder, "decoder": self.decoder}