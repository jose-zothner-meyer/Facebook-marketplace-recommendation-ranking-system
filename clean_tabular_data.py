import pandas as pd

class ProdClean:
    """
    Class for cleaning a tabular dataset containing product information.
    
    Key steps include:
      - Reading the CSV file with a specified line terminator.
      - Dropping unwanted columns (e.g., 'Unnamed: 0').
      - Removing rows with missing values.
      - Converting specific columns to string type.
      - Cleaning the 'price' column by removing currency symbols and commas, then converting to numeric.
      - Dropping rows with invalid prices.
    """
    
    def data_clean(self, input_file='Products.csv', line_terminator='\n'):
        """
        Perform data cleaning on the specified CSV file.
        
        Steps:
          1. Read CSV using the specified line terminator.
          2. Drop the 'Unnamed: 0' column if it exists.
          3. Drop rows with missing values.
          4. Convert selected columns to string type.
          5. Clean the 'price' column: remove '£' symbols and commas, convert to numeric.
          6. Drop rows where 'price' conversion fails (NaN).
          7. Return the cleaned DataFrame.
        
        :param input_file: Path to the CSV file to clean.
        :param line_terminator: Line terminator used in the CSV.
        :return: A cleaned pandas DataFrame.
        """
        # 1. Read the CSV file.
        df = pd.read_csv(input_file, lineterminator=line_terminator)
        
        # 2. Drop the 'Unnamed: 0' column if it exists.
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # 3. Drop rows with missing values.
        df.dropna(inplace=True)

        # 4. Convert selected columns to string.
        for col in ['id', 'product_name', 'category', 'product_description', 'location']:
            if col in df.columns:
                df[col] = df[col].astype('string')

        # 5. Clean the 'price' column.
        if 'price' in df.columns:
            df['price'] = df['price'].astype(str)  # Ensure it's a string.
            df['price'] = df['price'].str.replace('£', '', regex=False)  # Remove '£'.
            df['price'] = df['price'].str.replace(',', '', regex=False)  # Remove commas.
            df['price'] = pd.to_numeric(df['price'], errors='coerce')  # Convert to numeric.

            # 6. Drop rows where price conversion failed.
            df.dropna(subset=['price'], inplace=True)

        # 7. Return the cleaned DataFrame.
        return df