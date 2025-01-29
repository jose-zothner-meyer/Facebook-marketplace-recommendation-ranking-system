# clean_tabular_data.py

import pandas as pd

class ProdClean:
    """
    Class for cleaning a tabular dataset containing product information.
    """

    def data_clean(self, input_file='Products.csv', line_terminator='\n'):
        """
        Perform data cleaning on the specified CSV file.

        Steps:
        1. Read CSV with specified line terminator.
        2. Drop the 'Unnamed: 0' column if it exists.
        3. Drop rows with missing values (in any column).
        4. Convert columns to appropriate data types.
        5. Clean and convert the 'price' column to numeric.
        6. Drop rows that have invalid (NaN) prices.
        7. Return the cleaned DataFrame.

        :param input_file: Path to the CSV file to clean.
        :param line_terminator: Line terminator used in the CSV (default is '\\n').
        :return: A cleaned pandas DataFrame.
        """
        # 1. Read the CSV file using the specified line terminator
        df = pd.read_csv(input_file, lineterminator=line_terminator)
        
        # 2. Drop 'Unnamed: 0' if it exists (often created by certain export tools)
        if 'Unnamed: 0' in df.columns:
            df.drop('Unnamed: 0', axis=1, inplace=True)

        # 3. Drop rows with missing values in any column
        df.dropna(inplace=True)

        # 4. Convert certain columns to string data type (adjust as needed)
        for col in ['id', 'product_name', 'category', 'product_description', 'location']:
            if col in df.columns:
                df[col] = df[col].astype('string')

        # 5. Clean the 'price' column:
        #    - Remove '£' symbols
        #    - Remove commas (e.g., 1,000 -> 1000)
        #    - Convert to numeric
        if 'price' in df.columns:
            df['price'] = df['price'].astype(str)  # Ensure it's string before replacement
            df['price'] = df['price'].str.replace('£', '', regex=False)
            df['price'] = df['price'].str.replace(',', '', regex=False)
            df['price'] = pd.to_numeric(df['price'], errors='coerce')

            # 6. Drop rows where 'price' is NaN after conversion
            df.dropna(subset=['price'], inplace=True)

        # Return the cleaned DataFrame
        return df