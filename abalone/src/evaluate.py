"""
This module contains functions to evaluate the performance of the trained regression model for abalone age prediction.
log_history:
Initial description.
0. Fixed issue with handling invalid model files in load_model function.
0. Fixed issue with evaluate_model function not checking if the input model is a valid estimator with a predict method.
1. Fixed issue with evaluate_model function raising the wrong exception when the input model is not a valid estimator.

1. Import necessary libraries (sklearn.metrics).
2. Define load_model function:
    - Load model from specified file
    - Handle exceptions for non-existent model files and invalid model files
3. Define evaluate_model function:
    - Take model, X_test, and y_test as inputs
    - Make predictions on test data
    - Calculate metrics: Mean Squared Error (MSE), Mean Absolute Error (MAE), R-squared
    - Return dictionary of evaluation results
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import joblib
from sklearn.base import BaseEstimator

def load_model(model_path: str):
    """
    Load a pre-trained model from a specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        model: The loaded model object.

    Raises:
        FileNotFoundError: If the specified model file does not exist.
        ValueError: If the loaded model file is invalid or corrupted.
    """
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except Exception as e:
        raise ValueError(f"Invalid model file: {e}")
    return model


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the performance of a trained regression model on test data.

    Args:
        model: The trained regression model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test target values.

    Returns:
        dict: A dictionary containing the evaluation metrics:
            - 'mse': Mean Squared Error
            - 'mae': Mean Absolute Error
            - 'r2': R-squared score
    """
    if not isinstance(model, BaseEstimator):
        raise TypeError("Input model is not a valid estimator.")

    try:
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {'mse': mse, 'mae': mae, 'r2': r2}
    except Exception as e:
        raise ValueError(f"Error during model evaluation: {e}")