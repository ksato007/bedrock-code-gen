import pytest
import os
import sys
import tempfile
import shutil

def pytest_addoption(parser):
    parser.addini('venv_name', 'Name of the virtual environment', default='.venv')

def pytest_collection_modifyitems(items):
    # Ensure tests marked with 'last' run at the end of the test suite.
    last_tests = [item for item in items if item.get_closest_marker("last")]
    for test in last_tests:
        items.remove(test)
        items.append(test)

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment(request):
    # Set up the test environment before running tests.
    # Get the expected virtual environment path
    project_root = os.path.dirname(os.path.dirname(__file__))  # ex. breast_cancer directory
    expected_venv_path = os.path.join(project_root, "abalone_regression_venv")
    
    # Check if we're running in the correct virtual environment
    current_venv = os.environ.get("VIRTUAL_ENV")
    if current_venv != expected_venv_path:
        pytest.exit(f"Tests are not running in the expected virtual environment. Expected: {expected_venv_path} Current: {current_venv}")

    # Add the project root to sys.path if it's not already there
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Add the src directory to sys.path if it's not already there
    src_dir = os.path.join(project_root, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    # Print some debug information
    print(f"Project root: {project_root}")
    print(f"Expected venv path: {expected_venv_path}")
    print(f"Current VIRTUAL_ENV: {current_venv}")
    print(f"PYTHONPATH: {os.environ.get('PYTHONPATH', 'Not set')}")
    print(f"sys.path: {sys.path}")

    # Create a temporary directory for the test session
    temp_dir = tempfile.mkdtemp()
    request.addfinalizer(lambda: shutil.rmtree(temp_dir))
    return temp_dir

@pytest.fixture(scope="function")
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



@pytest.fixture(scope="function")
def temp_dir(tmp_path):
    # Provide a temporary directory for the test function.
    return tmp_path

@pytest.fixture(scope="function")
def temp_file(temp_dir):
    # Provide a temporary file for the test function.
    temp_file = temp_dir / "test_file.txt"
    temp_file.write_text("This is a test file.")
    return temp_file
