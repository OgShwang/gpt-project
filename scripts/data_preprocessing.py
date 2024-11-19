# data_preprocessing.py

import pandas as pd

def load_data():
    # Load dataset from CSV file
    df = pd.read_csv('data/dataset.csv')
    return df

def clean_data(df):
    # Handle missing values (e.g., fill with mean or drop)
    df = df.fillna(df.mean())  # Example: fill missing values with the mean of the column
    return df

def preprocess_data(df):
    # Example of preprocessing steps: scaling or encoding
    # If you have any categorical data, you can encode it here
    # Example: Encoding categorical variables if any
    # df = pd.get_dummies(df, drop_first=True)
    
    return df

# Example usage:
if __name__ == '__main__':
    data = load_data()
    clean_data = clean_data(data)
    preprocessed_data = preprocess_data(clean_data)
    print(preprocessed_data.head())
