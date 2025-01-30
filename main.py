"""
main.py

A top-level script to run all data processing tasks:
1. Data cleaning (via the ProdClean class from clean_tabular_data.py).
2. Product labeling (via the ProductLabeler class from product_labeler.py).
"""

from clean_tabular_data import ProdClean
from product_labeler import ProductLabeler
import pandas as pd

def main() -> None:
    """
    Main entry point for the data processing pipeline.

    This function performs two primary steps:
    1) Cleans the products data using the ProdClean class.
    2) Labels the cleaned data using the ProductLabeler class.

    It also demonstrates how to retrieve and print the encoder/decoder mappings
    after labeling the data.

    Steps:
    ------
    1) Initialize and run the data cleaning process (ProdClean) on 'data/Products.csv'.
       - The cleaned data is saved to 'data/Cleaned_Products.csv'.
    2) Initialize and run the labeling process (ProductLabeler) on the cleaned CSV.
       - The final labeled data is saved to 'data/Cleaned_Products_with_Labels.csv'.
    3) Print the encoder and decoder to show how categories map to labels and vice versa.
    """
    # 1. Clean the data
    cleaner = ProdClean()
    input_file = 'data/Products.csv'  # Original raw or uncleaned products file
    cleaned_file = 'data/Cleaned_Products.csv'

    try:
        print("Starting data cleaning process...")
        cleaned_df = cleaner.data_clean(input_file=input_file, line_terminator='\n')

        cleaned_df.to_csv(cleaned_file, index=False)
        print(f"Data cleaning complete! Cleaned file saved to: {cleaned_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")
        return

    # 2. Label the cleaned data
    labeled_file = 'data/training_data.csv'

    try:
        print("\nStarting labeling process...")
        labeler = ProductLabeler(
            products_file=cleaned_file,
            output_file=labeled_file
        )
        labeler.process()  # Runs the entire labeling pipeline

        # 3. Print encoder/decoder
        mappings = labeler.get_encoder_decoder()
        print("\nLabeling complete!")
        print("Encoder (Category -> Integer):", mappings["encoder"])
        print("Decoder (Integer -> Category):", mappings["decoder"])

    except FileNotFoundError:
        print(f"Error: The file '{cleaned_file}' was not found. Please check the file path.")
        return
    except Exception as e:
        print(f"An error occurred during labeling: {e}")
        return


if __name__ == "__main__":
    main()
