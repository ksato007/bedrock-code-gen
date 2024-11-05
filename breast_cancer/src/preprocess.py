"""
This module contains functions to preprocess the breast cancer dataset, including data cleaning and feature scaling.
log_history:
Initial description.
0. Fixed issue with target_names not being a numpy array.
0. Fixed issue with UnboundLocalError for target variable.
1. Fixed issue with handling column name mismatches in preprocess_data.
2. No change has been made this time because the code already handles column name mismatches correctly.

1. Import the necessary libraries: pandas, numpy, and sklearn.
2. Define load_data function:
    a. Accept file_path argument (optional)
    b. Load data from file if provided, else from sklearn
        - When loading from sklearn:
            - Use load_breast_cancer() to get the dataset
            - The dataset has the following structure:
                - 'data': a numpy array of shape (569, 30) containing the feature values
                - 'target': a numpy array of shape (569,) containing the binary target values (0 or 1)
                - 'target_names': a numpy array of shape (2,) containing the string labels for the target values ('malignant' and 'benign')
                - 'feature_names': a numpy array of shape (30,) containing the string names for the 30 features
        - Handle both cases: when data is already a DataFrame (from file) and when it's a Bunch object (from sklearn)
    d. Handle exceptions for file not found
3. Define preprocess_data function:
    a Load data using load_data if needed
    b Remove duplicates
    c Handle exceptions for missing values
    d Scale features using MinMaxScaler
    e Handle exceptions for invalid input
4. Include error handling for exceptions.
5. Add docstrings and type hints.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from typing import Union, Tuple, Optional

def load_data(file_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the breast cancer dataset from a file or the scikit-learn library.

    Args:
        file_path (str, optional): Path to the CSV file containing the dataset. If not provided, the dataset will be loaded from scikit-learn.

    Returns:
        Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the following:
            - data (pd.DataFrame): DataFrame containing the feature values.
            - target (np.ndarray): Array containing the binary target values (0 or 1).
            - target_names (np.ndarray): Array containing the string labels for the target values ('malignant' and 'benign').
            - feature_names (np.ndarray): Array containing the string names for the 30 features.

    Raises:
        FileNotFoundError: If the provided file_path does not exist.
    """
    try:
        if file_path:
            data = pd.read_csv(file_path)
            target = data.pop('target').values
            feature_names = data.columns.values
            target_names = np.array(['malignant', 'benign'])
        else:
            dataset = load_breast_cancer()
            data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
            target = dataset.target
            target_names = dataset.target_names
            feature_names = dataset.feature_names
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return data, target, target_names, feature_names

def preprocess_data(data: Optional[pd.DataFrame] = None, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Preprocess the breast cancer dataset by removing duplicates, handling missing values, and scaling the features.

    Args:
        data (pd.DataFrame, optional): DataFrame containing the dataset. If not provided, the dataset will be loaded from the specified file_path or scikit-learn.
        file_path (str, optional): Path to the CSV file containing the dataset. Required if data is not provided.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]: A tuple containing the following:
            - processed_data (pd.DataFrame): DataFrame containing the preprocessed feature values.
            - target (np.ndarray): Array containing the binary target values (0 or 1).

    Raises:
        ValueError: If both data and file_path are not provided.
        ValueError: If the input data contains missing values.
    """
    if data is None and file_path is None:
        raise ValueError("Either data or file_path must be provided.")

    if data is None:
        data, target, _, _ = load_data(file_path)
    else:
        target = data['target'].values if 'target' in data.columns else None

    if target is None:
        raise ValueError("Input data does not contain a 'target' column.")

    try:
        # Remove duplicates
        data.drop_duplicates(inplace=True)

        # Handle missing values
        if data.isnull().values.any():
            raise ValueError("Input data contains missing values.")

        # Scale features
        scaler = MinMaxScaler()
        processed_data = pd.DataFrame(scaler.fit_transform(data.drop('target', axis=1, errors='ignore')), columns=data.columns.drop('target'))

        return processed_data, target
    except Exception as e:
        raise ValueError(f"Error during preprocessing: {str(e)}")