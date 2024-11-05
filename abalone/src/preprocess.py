"""
This module contains functions to preprocess the abalone dataset, including data cleaning, encoding categorical variables, and feature scaling.
log_history:
Initial description.
0. Fixed issue with handling empty DataFrames in preprocess_data.
0. Fixed issue with handling missing values in preprocess_data.

1. Import the necessary libraries: pandas, numpy, and sklearn.
2. Define load_data function:
    a. Accept file_path argument (optional)
    b. Load data from file if provided, else from a public repository or local storage
    c. The dataset should have the following features:
       - 'Sex': categorical (M, F, I)
       - 'Length': continuous
       - 'Diameter': continuous
       - 'Height': continuous
       - 'Whole weight': continuous
       - 'Shucked weight': continuous
       - 'Viscera weight': continuous
       - 'Shell weight': continuous
       - 'Rings': integer (target variable, representing age)
    d. Handle exceptions for file not found
3. Define preprocess_data function:
    a. Load data using load_data if needed
    b. Handle missing values if any
    c. Encode the 'Sex' categorical variable (e.g., using one-hot encoding)
    d. Scale numerical features using StandardScaler
    e. Handle exceptions for invalid input
4. Include error handling for exceptions.
5. Add docstrings and type hints.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(file_path: str = None) -> pd.DataFrame:
    """
    Load the abalone dataset from a file or a public repository.

    Args:
        file_path (str, optional): Path to the dataset file. If not provided, the dataset will be loaded from a public repository or local storage.

    Returns:
        pd.DataFrame: The abalone dataset.

    Raises:
        FileNotFoundError: If the provided file path is invalid.
    """
    try:
        if file_path:
            data = pd.read_csv(file_path)
        else:
            # Load data from a public repository or local storage
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
            data = pd.read_csv(url, header=None, names=['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings'])
    except FileNotFoundError:
        raise FileNotFoundError("Invalid file path. Please provide a valid file path or use the default option.")

    return data


def preprocess_data(data: pd.DataFrame = None) -> tuple:
    """
    Preprocess the abalone dataset by handling missing values, encoding categorical variables, and scaling numerical features.

    Args:
        data (pd.DataFrame, optional): The abalone dataset. If not provided, the function will load the data using the load_data function.

    Returns:
        tuple: A tuple containing the preprocessed data (X) and the target variable (y).

    Raises:
        ValueError: If the input data is invalid or contains unexpected values.
    """
    try:
        if data is None:
            data = load_data()

        if data.empty:
            raise ValueError("Input data is empty.")

        # Handle missing values
        data = data.dropna()

        # Separate features and target
        X = data.drop('Rings', axis=1)
        y = data['Rings']

        # Encode categorical variable 'Sex'
        categorical_cols = ['Sex']
        numerical_cols = X.columns.drop(categorical_cols)

        categorical_transformer = OneHotEncoder(handle_unknown='ignore')
        numerical_transformer = StandardScaler()

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numerical_transformer, numerical_cols)
            ])

        pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
        X_preprocessed = pipeline.fit_transform(X)

        return X_preprocessed, y

    except ValueError as e:
        raise ValueError(f"Invalid input data: {e}")