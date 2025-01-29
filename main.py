from clean_tabular_data import ProdClean

def main():
    # Initialize the cleaner class
    cleaner = ProdClean()

    # Specify the input file and optional parameters
    input_file = 'Products.csv'  # Adjust this path to where your file is located
    output_file = 'Cleaned_Products.csv'  # Where you want to save the cleaned file

    # Run the cleaning process
    try:
        print("Starting data cleaning process...")
        cleaned_df = cleaner.data_clean(input_file=input_file, line_terminator='\n')

        # Save the cleaned DataFrame to a new file
        cleaned_df.to_csv(output_file, index=False)
        print(f"Data cleaning complete! Cleaned file saved to: {output_file}")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found. Please check the file path.")
    except Exception as e:
        print(f"An error occurred during data cleaning: {e}")

if __name__ == "__main__":
    main()
