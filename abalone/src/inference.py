"""
This module contains functions to make predictions on new, unseen data using the trained abalone age regression model.

1. Import necessary libraries (numpy, pandas) and from src.preprocess import preprocess_data.
2. Define a function `load_model` that loads the trained model from the specified file path:
    a. Use a try-except block to catch specific exceptions (FileNotFoundError, EOFError, etc.)
    b. Verify the loaded object is of the expected type (e.g., sklearn estimator or pipeline)
    c. If loading fails, raise a custom exception with a descriptive message
3. Define a function `preprocess_input_data` that takes new, unseen data as input.
4. Inside the `preprocess_input_data` function:
    a. Call the preprocess_data function from src.preprocess to preprocess the input data.
    b. Return the preprocessed input data.
    c. Implement error handling for exceptions if the input data is invalid or has incorrect data types.
5. Define a function `make_predictions` that takes the preprocessed input data and the loaded model as input, and returns the model predictions.
6. Inside the `make_predictions` function:
    a. Make predictions using the loaded model and the preprocessed input data.
    b. Implement error handling for exceptions when making predictions, such as invalid model or data.
    c. Return the predictions (abalone ages).
"""

import numpy as np
import pandas as pd
from src.preprocess import preprocess_data
import joblib

# src/inference.py

class ModelLoadingError(Exception):
    """Custom exception raised when the model fails to load."""
    pass

def load_model(model_path: str):
    """
    Load the trained model from the specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        The loaded model object.

    Raises:
        ModelLoadingError: If the model fails to load or is not of the expected type.
    """
    try:
        model = joblib.load(model_path)
    except (FileNotFoundError, EOFError) as e:
        raise ModelLoadingError(f"Failed to load the model: {e}")

    if not hasattr(model, 'predict'):
        raise ModelLoadingError("The loaded object is not a valid model.")

    return model

def preprocess_input_data(input_data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the input data for prediction.

    Args:
        input_data (pd.DataFrame): The new, unseen data to be preprocessed.

    Returns:
        pd.DataFrame: The preprocessed input data.

    Raises:
        ValueError: If the input data is invalid or has incorrect data types.
    """
    try:
        preprocessed_data = preprocess_data(input_data)
    except ValueError as e:
        raise ValueError(f"Invalid input data: {e}")

    return preprocessed_data

def make_predictions(preprocessed_data: pd.DataFrame, model) -> np.ndarray:
    """
    Make predictions using the loaded model and preprocessed input data.

    Args:
        preprocessed_data (pd.DataFrame): The preprocessed input data.
        model: The loaded model object.

    Returns:
        np.ndarray: The predicted abalone ages.

    Raises:
        ValueError: If the model or data is invalid for making predictions.
    """
    try:
        predictions = model.predict(preprocessed_data)
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Failed to make predictions: {e}")

    return predictions