import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to load and preprocess the data
def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Example: Fill missing values with the mean (if any)
    df.fillna(df.mean(), inplace=True)

    # Example: Encoding categorical variables (if needed)
    # Here we have Column2, which is categorical, so we’ll use label encoding as an example.
    df['Column2'] = df['Column2'].astype('category').cat.codes

    # Example: Feature scaling (normalization)
    scaler = StandardScaler()
    df[['Column1', 'Column3']] = scaler.fit_transform(df[['Column1', 'Column3']])

    # Split into features (X) and target (y)
    X = df[['Column1', 'Column2']]  # Features
    y = df['Column3']  # Target variable

    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

# If you want to test the preprocessing function:
# X_train, X_test, y_train, y_test = preprocess_data('/content/gpt-project/data/dataset.csv')
