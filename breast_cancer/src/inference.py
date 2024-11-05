"""
This module contains functions to make predictions on new, unseen data using the trained binary classification model for the breast cancer dataset.
log_history:
Initial description.
0. Fixed issue with returning target variable from preprocess_data function.
1. No change has been made this time because the issue is in the src/preprocess.py file, which should not be modified.
2. No change has been made this time because the issue is in the src/preprocess.py file, which should not be modified.

1. Import necessary libraries (numpy, pandas, tensorflow) and from src.preprocess import preprocess_data.
2. Define a function `load_model` that loads the trained model from the specified file path. The model should be saved in the TensorFlow SavedModel format or the Keras V3 format 
3. Define a function `preprocess_input_data` that takes new, unseen data as input.
4. Inside the `preprocess_input_data` function:
    a. Call the preprocess_data function from src.preprocess to preprocess the input data, including handling missing values and scaling the features.
    b. Return the preprocessed input data.
    c. Implement error handling for exceptions if the input data is invalid or has incorrect data types.
5. Define a function `make_predictions` that takes the preprocessed input data and the loaded model as input, and returns the model predictions.
6. Inside the `make_predictions` function:
    a. Make predictions using the loaded model and the preprocessed input data.
    b. Implement error handling for exceptions when making predictions, such as invalid model or data.
    c. Return the predictions.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from src.preprocess import preprocess_data

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load the trained model from the specified file path.

    Args:
        model_path (str): Path to the saved model file.

    Returns:
        tf.keras.Model: The loaded model.

    Raises:
        Exception: If the model file cannot be loaded.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")


def preprocess_input_data(input_data: pd.DataFrame) -> np.ndarray:
    """
    Preprocess the input data for prediction.

    Args:
        input_data (pd.DataFrame): The input data to be preprocessed.

    Returns:
        np.ndarray: The preprocessed input data.

    Raises:
        ValueError: If the input data is invalid or has incorrect data types.
    """
    try:
        preprocessed_data = preprocess_data(input_data)[0]
        return preprocessed_data
    except ValueError as e:
        raise ValueError(f"Error preprocessing input data: {e}")


def make_predictions(input_data: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    """
    Make predictions using the loaded model and preprocessed input data.

    Args:
        input_data (np.ndarray): The preprocessed input data.
        model (tf.keras.Model): The loaded model.

    Returns:
        np.ndarray: The model predictions.

    Raises:
        Exception: If an error occurs during prediction.
    """
    try:
        predictions = model.predict(input_data)
        return predictions
    except Exception as e:
        raise Exception(f"Error making predictions: {e}")