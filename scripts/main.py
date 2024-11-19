# main.py

import numpy as np
from data_preprocessing import load_data, clean_data, preprocess_data
from model import create_model
from tensorflow.keras.callbacks import ModelCheckpoint

def train_model():
    # Load and preprocess data
    data = load_data()
    clean_data = clean_data(data)
    preprocessed_data = preprocess_data(clean_data)

    # Split data into features (X) and target (y)
    X = preprocessed_data.iloc[:, :-1].values  # Assuming last column is the target
    y = preprocessed_data.iloc[:, -1].values

    # Create the model
    model = create_model(input_shape=X.shape[1])

    # Train the model
    checkpoint = ModelCheckpoint('models/trained_model.h5', save_best_only=True)
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

    # Save the trained model
    model.save('models/trained_model.h5')

if __name__ == '__main__':
    train_model()
