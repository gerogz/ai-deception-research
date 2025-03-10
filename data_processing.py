import pandas as pd

def load_and_clean_data(file_path):
    """
    Load CSV file, filter out rows with invalid dates, and return cleaned DataFrame.
    
    Parameters:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with only relevant columns.
    """
    data = pd.read_csv(file_path)

    # Filter out invalid dates in 'StartDate'
    data = data[pd.to_datetime(data['StartDate'], errors='coerce').notna()]

    # Filter relevant risk columns
    risk_columns = [col for col in data.columns if '_selfRisk' in col or '_otherRisk' in col]
    lie_type_columns = [col for col in data.columns if '_lieType' in col]
    relevant_columns = risk_columns + lie_type_columns

    # Keep only necessary columns
    data_cleaned = data[relevant_columns].copy()

    # Convert risk columns to numeric
    for col in risk_columns:
        data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')

    return data_cleaned
