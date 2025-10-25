"""
Script to remove redundant columns (shape_file, name, class_b) from the CSV file.
The last three columns are redundant since:
- 'shape_file' is the same as 'shape'
- 'name' is the same as 'shape'
- 'class_b' is the same as 'class'
"""

import pandas as pd
import os

def remove_redundant_columns(input_csv, output_csv=None):
    """
    Remove the last three columns (shape_file, name, class_b) from the CSV file.
    
    Args:
        input_csv (str): Path to input CSV file
        output_csv (str): Path to output CSV file (if None, overwrites input)
    """
    # Read the CSV file
    print(f"Reading CSV file: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Display original shape
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Remove the last three columns
    columns_to_drop = df.columns[-3:].tolist()
    print(f"\nRemoving columns: {columns_to_drop}")
    
    df_cleaned = df.iloc[:, :-3]
    
    # Display new shape
    print(f"New shape: {df_cleaned.shape}")
    print(f"New columns: {list(df_cleaned.columns)}")
    
    # Save the cleaned dataframe
    if output_csv is None:
        output_csv = input_csv
    
    df_cleaned.to_csv(output_csv, index=False)
    print(f"\nCleaned CSV saved to: {output_csv}")
    
    return df_cleaned


if __name__ == "__main__":
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    input_file = os.path.join(script_dir, "final_006_cleaned.csv")
    output_file = os.path.join(script_dir, "final_006_cleaned_v2.csv")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File not found: {input_file}")
        exit(1)
    
    # Remove redundant columns
    remove_redundant_columns(input_file, output_file)
    
    print("\nDone! The CSV file has been updated.")
