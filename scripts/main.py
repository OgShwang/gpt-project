from data_preprocessing import preprocess_data
from model import train_model

# Load and preprocess the data
X_train, X_test, y_train, y_test = preprocess_data('/content/gpt-project/data/dataset.csv')

# Train the model and evaluate it
model, mse = train_model(X_train, X_test, y_train, y_test)

# Print out the model's Mean Squared Error
print(f'Model training completed with MSE: {mse}')
