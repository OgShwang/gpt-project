from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Function to train the model
def train_model(X_train, X_test, y_train, y_test):
    # Initialize the model (Linear Regression for now)
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')

    # Save the trained model to a file
    joblib.dump(model, '/content/gpt-project/models/trained_model.h5')

    return model, mse

# To train the model, you would call:
# model, mse = train_model(X_train, X_test, y_train, y_test)
