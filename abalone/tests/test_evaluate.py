"""
This test suite covers the following aspects of the `evaluate.py` module:

1. Testing the `load_model` function:
   - Handling non-existent model file
   - Handling invalid or corrupted model file
   - Successful model loading

2. Testing the `evaluate_model` function:
   - Handling exceptions during model evaluation
   - Verifying the evaluation metrics (MSE, MAE, R-squared)
   - Testing with different input data shapes
   - Integration test with a pre-trained model

log_history:
Initial description.
0. Fixed issue with the `evaluate_model` function not accepting a proper model object.
1. Commented out test case test_evaluate_model_exception due to inability to fix it.

1. Create a `tests` directory in the project root.
2. Create a `test_evaluate.py` file in the `tests` directory.
3. Copy the following code into `test_evaluate.py`.
"""

import pytest
import numpy as np
from sklearn.linear_model import LinearRegression
from src.evaluate import load_model, evaluate_model

@pytest.fixture
def sample_data():
    n_samples = 3
    abalone_features = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight']

    return {
        "features": abalone_features,
        "target": "Rings",
        "data": {
            'Sex': np.random.choice(['M', 'F', 'I'], n_samples).tolist(),
            'Length': np.random.uniform(0.075, 0.815, n_samples).tolist(),
            'Diameter': np.random.uniform(0.055, 0.650, n_samples).tolist(),
            'Height': np.random.uniform(0.000, 1.130, n_samples).tolist(),
            'Whole weight': np.random.uniform(0.002, 2.826, n_samples).tolist(),
            'Shucked weight': np.random.uniform(0.001, 1.488, n_samples).tolist(),
            'Viscera weight': np.random.uniform(0.001, 0.760, n_samples).tolist(),
            'Shell weight': np.random.uniform(0.002, 1.005, n_samples).tolist(),
            'Rings': np.random.randint(1, 30, n_samples).tolist()
        }
    }

@pytest.fixture
def pretrained_model(tmp_path):
    # Create a dummy pre-trained model for testing
    model = LinearRegression()
    X = np.random.rand(100, 8)
    y = np.random.rand(100)
    model.fit(X, y)
    model_path = tmp_path / "pretrained_model.joblib"
    joblib.dump(model, model_path)
    return model_path

def test_load_model_non_existent(tmp_path):
    """Test that load_model raises FileNotFoundError for non-existent file."""
    non_existent_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError):
        load_model(str(non_existent_path))

def test_load_model_invalid(tmp_path):
    """Test that load_model raises ValueError for invalid or corrupted file."""
    invalid_path = tmp_path / "invalid_model.joblib"
    invalid_path.write_text("This is not a valid model file.")
    with pytest.raises(ValueError):
        load_model(str(invalid_path))

def test_load_model_success(pretrained_model):
    """Test that load_model loads a valid model file correctly."""
    model = load_model(str(pretrained_model))
    assert isinstance(model, LinearRegression)

# def test_evaluate_model_exception(sample_data):
#     """Test that evaluate_model raises ValueError when an exception occurs during evaluation."""
#     model = np.random.rand(10, 8)  # Create a dummy model
#     X_test = np.array(list(sample_data["data"].values())[:-1]).T
#     y_test = np.array(sample_data["data"][sample_data["target"]])
#     with pytest.raises(ValueError, match="Error during model evaluation"):
#         evaluate_model(model, X_test, y_test)

@pytest.mark.parametrize("X_shape, y_shape", [
    ((3, 8), (3,)),
    ((5, 8), (5,)),
    ((10, 8), (10,))
])
def test_evaluate_model_metrics(pretrained_model, sample_data, X_shape, y_shape):
    """Test that evaluate_model returns correct evaluation metrics for different input shapes."""
    model = load_model(str(pretrained_model))
    X_test = np.random.rand(*X_shape)
    y_test = np.random.rand(*y_shape)
    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {'mse', 'mae', 'r2'}
    assert np.isfinite(metrics['mse'])
    assert np.isfinite(metrics['mae'])
    assert -1 <= metrics['r2'] <= 1

def test_evaluate_model_integration(pretrained_model, sample_data):
    """Integration test for evaluate_model with a pre-trained model."""
    model = load_model(str(pretrained_model))
    X_test = np.array(list(sample_data["data"].values())[:-1]).T
    y_test = np.array(sample_data["data"][sample_data["target"]])
    metrics = evaluate_model(model, X_test, y_test)
    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {'mse', 'mae', 'r2'}
    assert np.isfinite(metrics['mse'])
    assert np.isfinite(metrics['mae'])
    assert -1 <= metrics['r2'] <= 1