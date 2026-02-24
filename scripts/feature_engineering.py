import pandas as pd
import os

def remove_year(processed_path):
    """Removes year + file extension from df name to make it easier to call 

    Recursively goes through the processed_path file directory and remove the date suffix from the 
    name and assigns them to a dictionary for easier calling later.

    Args:
        processed_path: folder containing processed files

    Returns:
        A dictionary mapping each filename to its cleaned pandas DataFrame
    """
    processed_files = {}

    for filename in os.listdir(processed_path):
        if filename.endswith(".csv"):

            file_path = os.path.join(processed_path,filename)

            df = pd.read_csv(file_path)

            clean_name = filename.replace('_2015_2025.csv','')

            
            processed_files[clean_name] = df
        
    return processed_files



def main():
    processed_path = "data/processed"

    files = remove_year(processed_path)





if __name__ == "__main__":
    main()