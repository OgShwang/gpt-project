from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  # This is the Random Forest model we need to import
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Function to train and evaluate the Linear Regression model
def train_model(X_train, X_test, y_train, y_test):
    # Initialize the model (Linear Regression for now)
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

    # Save the trained model to a file
    joblib.dump(model, '/content/gpt-project/models/trained_model.h5')

    return model, mse, mae, r2


# Function to train and evaluate the Random Forest model
def train_random_forest(X_train, X_test, y_train, y_test):
    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)  # 100 trees

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'Mean Absolute Error: {mae}')
    print(f'R² Score: {r2}')

    # Save the trained model
    joblib.dump(model, '/content/gpt-project/models/random_forest_model.h5')

    return model, mse, mae, r2
