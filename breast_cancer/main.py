"""
This module serves as the entry point for the breast cancer classification program.

1. Import necessary functions and modules:
    a. From src.preprocess import load_data and preprocess_data.
    b. From src.model import split_data and train_model.
    c. From src.evaluate import load_model and evaluate_model.
    d. From src.inference import preprocess_input_data and make_predictions.
    e. Import argparse for command-line argument handling, os for file path handling, and pandas and numpy for data manipulation.
2. Define the main function.
3. Inside the main function:
    a. Set up argument parsing for data_path, model_path, epochs, and batch_size.
        - data_path: str, default=""
        - model_path: str, default="trained_model_1.keras"
        - epochs: int, default=100
        - batch_size: int, default=32
    b. Use a try-except block to handle potential exceptions.
    c. Load the data from the specified data_path using the load_data function from the src.preprocess module. The load_data function should return the following:
        - X: A pandas DataFrame containing the features.
        - y: A pandas Series named 'target' containing the target variable.
        - target_names: A list or array containing the names of the target classes.
        - feature_names: A list or array containing the names of the features.
    d. Ensure that the loaded data is in the correct format:
        - The features (X) should be a pandas DataFrame.
        - The target (y) should be a pandas Series and rename the target variable (y) to 'target'.
    e. Print the shapes of the original data.
    f. Preprocess the data using preprocess_data from preprocess.py.
    g. Print the shapes of the preprocessed data.
    h. Split the data using split_data from model.py.
        - X_train (np.ndarray): The training input data.
        - X_val (np.ndarray): The validation input data.
        - X_test (np.ndarray): The test input data.
        - y_train (np.ndarray): The training target data.
        - y_val (np.ndarray): The validation target data.
        - y_test (np.ndarray): The test target data.
    i. Train the model using train_model from model.py.
    j. If model training is successful:
        - Load the trained model using load_model from evaluate.py.
        - Evaluate the model using evaluate_model from evaluate.py.
        - Print evaluation results.
        - Demonstrate inference on a small subset of test data.
    k. Print target names and feature names.
4. Implement comprehensive error handling to catch and print any exceptions.
5. Use if __name__ == "__main__": to call the main function when the script is run directly.
"""

import argparse
import numpy as np
import pandas as pd
from src.preprocess import load_data, preprocess_data
from src.model import split_data, train_model
from src.evaluate import load_model, evaluate_model
from src.inference import preprocess_input_data, make_predictions

def main(data_path: str, model_path: str, epochs: int, batch_size: int) -> None:
    """
    Main function to run the breast cancer classification program.

    Args:
        data_path (str): Path to the data file.
        model_path (str): Path to save the trained model.
        epochs (int): Number of epochs for model training.
        batch_size (int): Batch size for model training.

    Returns:
        None
    """
    try:
        # Load data
        X, y, target_names, feature_names = load_data(data_path)
        y = y.rename('target', inplace=True)

        print(f"Original data shapes: X: {X.shape}, y: {y.shape}")

        # Preprocess data
        X, y = preprocess_data(X, y)

        print(f"Preprocessed data shapes: X: {X.shape}, y: {y.shape}")

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

        # Train model
        model = train_model(X_train, y_train, epochs, batch_size)

        if model:
            # Load trained model
            loaded_model = load_model(model_path)

            # Evaluate model
            evaluation_metrics = evaluate_model(loaded_model, X_test, y_test)
            print(f"Evaluation metrics: {evaluation_metrics}")

            # Demonstrate inference
            test_sample = X_test[:5]
            preprocessed_sample = preprocess_input_data(test_sample)
            predictions = make_predictions(loaded_model, preprocessed_sample)
            print(f"Sample predictions: {predictions}")

        print(f"Target names: {target_names}")
        print(f"Feature names: {feature_names}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Breast Cancer Classification")
    parser.add_argument("--data_path", type=str, default="", help="Path to the data file")
    parser.add_argument("--model_path", type=str, default="trained_model_1.keras", help="Path to save the trained model")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs for model training")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for model training")
    args = parser.parse_args()

    main(args.data_path, args.model_path, args.epochs, args.batch_size)