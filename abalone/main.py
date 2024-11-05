"""
This module serves as the entry point for the abalone age regression program.

1. Import necessary functions and modules:
    a. From src.preprocess import load_data and preprocess_data.
    b. From src.model import split_data, train_model, and save_model.
    c. From src.evaluate import load_model and evaluate_model.
    d. From src.inference import preprocess_input_data and make_predictions.
    e. Import argparse for command-line argument handling, os for file path handling, and pandas and numpy for data manipulation.
2. Define the main function.
3. Inside the main function:
    a. Set up argument parsing for data_path and model_path.
        - data_path: str, default=""
        - model_path: str, default="trained_model.joblib"
    b. Use a try-except block to handle potential exceptions.
    c. Load the data from the specified data_path using the load_data function from the src.preprocess module.
    d. Preprocess the data using preprocess_data from preprocess.py.
    e. Split the data using split_data from model.py.
    f. Train the model using train_model from model.py.
    g. Save the trained model using save_model from model.py.
    h. Evaluate the model using evaluate_model from evaluate.py.
    i. Print evaluation results.
    j. Demonstrate inference on a small subset of test data.
4. Implement comprehensive error handling to catch and print any exceptions.
5. Use if __name__ == "__main__": to call the main function when the script is run directly.
"""

import argparse
import numpy as np
import pandas as pd
from src.preprocess import load_data, preprocess_data
from src.model import split_data, train_model, save_model
from src.evaluate import load_model, evaluate_model
from src.inference import preprocess_input_data, make_predictions

def main(data_path: str, model_path: str) -> None:
    """
    Main function to run the abalone age regression pipeline.

    Args:
        data_path (str): Path to the data file.
        model_path (str): Path to save the trained model.

    Returns:
        None
    """
    try:
        # Load data
        data = load_data(data_path)

        # Preprocess data
        X, y = preprocess_data(data)

        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Train model
        model = train_model(X_train, y_train)

        # Save model
        save_model(model, model_path)

        # Evaluate model
        loaded_model = load_model(model_path)
        evaluation_metrics = evaluate_model(loaded_model, X_test, y_test)
        print(f"Evaluation Metrics: {evaluation_metrics}")

        # Demonstrate inference
        sample_data = X_test.sample(5)
        sample_predictions = make_predictions(loaded_model, preprocess_input_data(sample_data))
        print(f"Sample Predictions: {sample_predictions}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Abalone Age Regression")
    parser.add_argument("--data_path", type=str, default="", help="Path to the data file")
    parser.add_argument("--model_path", type=str, default="trained_model.joblib", help="Path to save the trained model")
    args = parser.parse_args()

    main(args.data_path, args.model_path)