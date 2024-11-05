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
    expected_venv_path = os.path.join(project_root, "breast_cancer_venv")
    
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
    # Provide sample data for tests.
    # Return some sample data that can be used across multiple tests
    return {
        "features": [f"feature_{i}" for i in range(30)],
        "target": "diagnosis",
        "data": {
            f"feature_{i}": [1.0, 2.0, 3.0] for i in range(30)
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
