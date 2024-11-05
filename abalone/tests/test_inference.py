"""
This set of test cases covers various aspects of the `src/inference.py` module, including loading the trained model, preprocessing input data, making predictions, and handling errors and edge cases. The tests utilize fixtures for sample data and temporary files, flexible floating-point comparisons, exception handling, parameterization, and integration testing.

1. Test the `load_model` function:
   - Test successful loading of a valid model file
   - Test handling of FileNotFoundError when the model file is not found
   - Test handling of EOFError when the model file is corrupted or empty
   - Test handling of an invalid model object (without the `predict` method)

2. Test the `preprocess_input_data` function:
   - Test successful preprocessing of valid input data
   - Test handling of ValueError when the input data is invalid or has incorrect data types

3. Test the `make_predictions` function:
   - Test successful prediction with valid input data and model
   - Test handling of ValueError when the model or data is invalid for making predictions
   - Test handling of AttributeError when the model object does not have the `predict` method

4. Implement an integration test to verify the end-to-end functionality of the module:
   - Load a valid model
   - Preprocess sample input data
   - Make predictions using the loaded model and preprocessed data
   - Verify the predictions against expected values

5. Use parameterization to test the `make_predictions` function with multiple input data samples.
"""

import pytest
import numpy as np
from src.inference import load_model, preprocess_input_data, make_predictions, ModelLoadingError

# Test cases for the load_model function
def test_load_model_success(tmp_path):
    # Create a temporary model file
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy model data")

    # Test successful loading of a valid model file
    model = load_model(str(model_file))
    assert model is not None

def test_load_model_file_not_found():
    # Test handling of FileNotFoundError
    with pytest.raises(ModelLoadingError, match="Failed to load the model"):
        load_model("nonexistent_file.joblib")

def test_load_model_eof_error(tmp_path):
    # Create an empty temporary file
    empty_file = tmp_path / "empty_file.joblib"
    empty_file.touch()

    # Test handling of EOFError
    with pytest.raises(ModelLoadingError, match="Failed to load the model"):
        load_model(str(empty_file))

def test_load_model_invalid_object(tmp_path):
    # Create a temporary file with invalid data
    invalid_file = tmp_path / "invalid_file.joblib"
    invalid_file.write_text("This is not a valid model object.")

    # Test handling of an invalid model object
    with pytest.raises(ModelLoadingError, match="not a valid model"):
        load_model(str(invalid_file))

# Test cases for the preprocess_input_data function
def test_preprocess_input_data_success(sample_data):
    # Test successful preprocessing of valid input data
    input_data = pd.DataFrame(sample_data["data"])
    preprocessed_data = preprocess_input_data(input_data)
    assert preprocessed_data is not None

def test_preprocess_input_data_invalid_data():
    # Test handling of ValueError when the input data is invalid
    invalid_data = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="Invalid input data"):
        preprocess_input_data(invalid_data)

# Test cases for the make_predictions function
def test_make_predictions_success(sample_data, tmp_path):
    # Create a temporary model file
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy model data")

    # Load the model
    model = load_model(str(model_file))

    # Preprocess the input data
    input_data = pd.DataFrame(sample_data["data"])
    preprocessed_data = preprocess_input_data(input_data)

    # Test successful prediction with valid input data and model
    predictions = make_predictions(preprocessed_data, model)
    assert predictions is not None

def test_make_predictions_invalid_model(sample_data):
    # Test handling of ValueError when the model is invalid
    input_data = pd.DataFrame(sample_data["data"])
    preprocessed_data = preprocess_input_data(input_data)
    invalid_model = "This is not a valid model object."

    with pytest.raises(ValueError, match="Failed to make predictions"):
        make_predictions(preprocessed_data, invalid_model)

def test_make_predictions_invalid_data(sample_data, tmp_path):
    # Create a temporary model file
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy model data")

    # Load the model
    model = load_model(str(model_file))

    # Test handling of ValueError when the data is invalid
    invalid_data = pd.DataFrame({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    with pytest.raises(ValueError, match="Failed to make predictions"):
        make_predictions(invalid_data, model)

# Integration test
def test_integration(sample_data, tmp_path):
    # Create a temporary model file
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy model data")

    # Load the model
    model = load_model(str(model_file))

    # Preprocess the input data
    input_data = pd.DataFrame(sample_data["data"])
    preprocessed_data = preprocess_input_data(input_data)

    # Make predictions using the loaded model and preprocessed data
    predictions = make_predictions(preprocessed_data, model)

    # Verify the predictions against expected values
    expected_predictions = np.array([10, 15, 20])
    assert np.allclose(predictions, expected_predictions, rtol=1e-5, atol=1e-5)

# Test with multiple input data samples using parameterization
@pytest.mark.parametrize("input_data", [
    pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]}),
    pd.DataFrame({"A": [10, 20, 30], "B": [40, 50, 60]}),
    pd.DataFrame({"A": [100, 200, 300], "B": [400, 500, 600]})
])
def test_make_predictions_multiple_inputs(input_data, tmp_path):
    # Create a temporary model file
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"dummy model data")

    # Load the model
    model = load_model(str(model_file))

    # Preprocess the input data
    preprocessed_data = preprocess_input_data(input_data)

    # Make predictions using the loaded model and preprocessed data
    predictions = make_predictions(preprocessed_data, model)

    # Verify that predictions are not None
    assert predictions is not None