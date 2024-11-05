from .graph_state import SharedState, TestType, FileState
from .utils import FileHandler
from .logger import setup_logging, log

setup_logging("INFO")

class FileGenerator:
    """
    The FileGenerator class is responsible for generating source code and test files from FileState
    objects, which contain the file content and metadata, and writing them to disk using the FileHandler utility
    """
    @staticmethod
    def create_files_from_state(state: SharedState):
        """
        Creates source and test files from the provided SharedState object.

        Args:
            state (SharedState): The SharedState object containing the FileState objects.
        """
        FileGenerator.create_source_files(state)
        FileGenerator.create_test_files(state)

    @staticmethod
    def create_source_files(state: SharedState):
        """
        Creates source files from the FileState objects in the SharedState object.

        Args:
            state (SharedState): The SharedState object containing the FileState objects.
        """
        for file_state in state.source_file_states.values():
            FileGenerator.create_file_from_state(file_state)

    @staticmethod
    def create_test_files(state: SharedState):
        """
        Creates test files from the FileState objects in the SharedState object.

        Args:
            state (SharedState): The SharedState object containing the FileState objects.
        """
        for file_state in state.test_file_states.values():
            FileGenerator.create_file_from_state(file_state)

    @staticmethod
    def create_file_from_state(file_state: FileState):
        """
        Creates a file from the provided FileState object.

        Args:
            file_state (FileState): The FileState object containing the file content and metadata.
        """
        content = file_state.content
        filename = file_state.filename
        file_type = "python" if filename.endswith(".py") else "text"
        # Determine the file type based on the file extension
        # Special handling for requirements.txt file
        if filename == "requirements.txt":
            content = content.strip().replace("==", ">=")
        # Create a dictionary with the file data
        file_data = {
            "content": content,
            "filename": filename,
            "file_type": file_type
        }
        FileHandler.create_file(file_data=file_data)
    
    @staticmethod
    def create_ini_file(dir_name: str = ''):
        FileHandler.create_ini_file(dir_name)
