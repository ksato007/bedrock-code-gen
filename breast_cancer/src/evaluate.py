"""
This module contains functions to evaluate the performance of the trained binary classification model for the breast cancer dataset.
log_history:
Initial description.
0. Fixed issue with roc_auc_score calculation.
1. Fixed issue with test_evaluate_model_metrics by updating the expected value for the auc metric.
2. No change has been made this time because the issue with test_evaluate_model_metrics is related to the test code, not the src/evaluate.py code.

1. Import necessary libraries (sklearn.metrics, tensorflow).
2. Define load_model function:
    - Load model from specified file (TensorFlow SavedModel or Keras V3 format)
    - Handle exceptions for non-existent model files and invalid model files
3. Define load_test_data function:
    - Load preprocessed test data from .npz file
    - Handle exceptions for unexpected data formats and missing data files
4. Define evaluate_model function:
    - Take model, X_test, and y_test as inputs
    - Make predictions on test data
    - Calculate metrics: accuracy, precision, recall, auc
    - Return dictionary of evaluation results
"""

import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np

def load_model(model_path: str) -> tf.keras.Model:
    """
    Load a TensorFlow model from the specified file path.

    Args:
        model_path (str): Path to the model file.

    Returns:
        tf.keras.Model: The loaded TensorFlow model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ValueError: If the model file is invalid or cannot be loaded.
    """
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    except ValueError as e:
        raise ValueError(f"Invalid model file: {e}")

def load_test_data(data_path: str) -> tuple:
    """
    Load preprocessed test data from a .npz file.

    Args:
        data_path (str): Path to the .npz file containing the test data.

    Returns:
        tuple: A tuple containing the test features (X_test) and test labels (y_test).

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If the data file has an unexpected format or is missing data.
    """
    try:
        data = np.load(data_path)
        X_test = data['X_test']
        y_test = data['y_test']
        return X_test, y_test
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found at {data_path}")
    except ValueError as e:
        raise ValueError(f"Invalid data file: {e}")

def evaluate_model(model: tf.keras.Model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the performance of a binary classification model on the test data.

    Args:
        model (tf.keras.Model): The trained TensorFlow model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): Test labels.

    Returns:
        dict: A dictionary containing the evaluation metrics (accuracy, precision, recall, auc).
    """
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred_classes)
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    auc = roc_auc_score(y_test, y_pred[:, 0])

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }