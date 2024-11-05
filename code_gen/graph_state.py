from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from .code_models import Code
from .logger import setup_logging, log

setup_logging("INFO")

class FileStatus(Enum):
    IMPORT_TEST_PENDING = 1
    IMPORT_TEST_FAILED = 2
    IMPORT_TEST_PASSED = 3
    CODE_TEST_PENDING = 4
    CODE_TEST_FAILED = 5
    CODE_TEST_PASSED = 6
    PYTEST_PENDING = 7
    PYTEST_FAILED = 8
    PYTEST_PASSED = 9
    ALL_TESTS_PASSED = 10

class TestType(Enum):
    IMPORT = 1
    CODE = 2
    PYTEST = 3

class FileStateConstants:
    DEFAULT_TEST_PASSED = False
    DEFAULT_ITERATION_COUNT = 0
    DEFAULT_MAX_ITERATIONS = 10

class FileType(Enum):
    SOURCE = 1
    TEST = 2

class FileState(BaseModel):
    """
    The FileState class represents the state of a file during the code generation process. 
    It keeps track of various attributes such as the filename, file type (source or test), 
    solution (a dictionary representing the code), content (the actual code string), test results (import, code, and pytest), iteration count, maximum iterations, 
    messages (a list of tuples containing messages), status (an enum representing the current status of the file), and error (if any). 
    The class also provides methods to initialize file states from baseline code, update the state with new code, handle tests, run tests, and set the status. 
    Additionally, it defines several helper enums and constants related to file status, test types, and default values.
    """
    filename: str
    file_type: FileType
    solution: Dict[str, str] = Field(default_factory=dict) # need to check imports and code separately
    content: str = '' # need it to write to file
    import_passed: bool = FileStateConstants.DEFAULT_TEST_PASSED
    code_passed: bool = FileStateConstants.DEFAULT_TEST_PASSED
    pytest_passed: bool = FileStateConstants.DEFAULT_TEST_PASSED
    iteration_count: int = FileStateConstants.DEFAULT_ITERATION_COUNT
    max_iterations: int = FileStateConstants.DEFAULT_MAX_ITERATIONS
    messages: List[Any] = Field(default_factory=list)
    status: FileStatus = FileStatus.IMPORT_TEST_PENDING
    error: Optional[str] = ''

    @classmethod
    def initialize_file_states_from_baseline_code(cls, codes: 'BaselineCode', messages: List[Tuple[str, str]]) -> Dict[str, 'FileState']: # type: ignore
        """
        Creates a dictionary of file states based on the provided codes and messages.

        Args:
            codes (Union[BaselineCode, TestCode]): The codes to be processed.
            messages (List[Tuple[str, str]]): A list of messages to be included in the file states.

        Returns:
            Dict[str, FileState]: A dictionary of file states, where the keys are the filenames and the values are the corresponding FileState objects.
        """
        file_states = {}
        # is it better to append all files or each file separately?
        message = "Here is the suggested solution for {filename}:\n\n{code_string}"
        #message = f"Here is the suggested solution:\n\n{codes.__dict__}"
        for filename, file_content in codes.__fields__.items():
            if file_content.type_ == Code:
                code_instance = getattr(codes, filename)
                file_states[filename] = FileState(
                    filename=code_instance.filename,
                    file_type= FileType.PYTEST if code_instance.filename.startswith('tests/') or code_instance.filename.startswith('test_') else FileType.SOURCE,
                    solution=code_instance.dict(exclude_none=True),
                    content=code_instance.to_string(),
                    iteration_count=1,
                    # is it better to append all files or each file separately?
                    messages=messages+[(
                        "assistant",
                        message.format(filename=code_instance.filename,
                                       code_string=code_instance.to_string())
                    )]
                    # messages=messages+[("assistant",message)]
                )
        return file_states

    def update_state_from_code(self, code: 'Code', max_messages: int = 4)->'FileState':
        """
        Updates the file state with the provided code.

        Args:
            code (Code): The new code to be used to update the file state.
            max_messages: max num messages to retain
        Returns:
            FileState: The updated file state.
        """
        message = "Here is the suggested solution for {filename}:\n\n{code_string}"
        self.solution = code.dict(exclude_none=True)
        self.content = code.to_string()
        self.iteration_count += 1
        # self.messages.append((
        #     "assistant",
        #     message.format(filename=code.filename, code_string=code.to_string())
        # ))
        new_message = (
            "assistant",
            message.format(filename=code.filename, code_string=code.to_string())
        )
        self.update_messages(new_message=new_message, max_messages=max_messages)
        self.error = ''

        return self
    
    def set_all_status_passed(self):
        self.set_status(FileStatus.ALL_TESTS_PASSED)
        self.import_passed = True
        self.code_passed = True
        self.pytest_passed = True

    def handle_test(self, test_type: TestType, venv_name=str)->'FileState':
        
        status, has_passed, error_message, code_string = self.run_test(test_type, venv_name)
        log(f"Status: {status}, Has Passed: {has_passed}, Error Message: {error_message}, Code String: {code_string}", "DEBUG")
        self.error = error_message
        # Update the file status
        self.set_status(status)

        # Set the individual test result attribute
        setattr(self, f"{test_type.name.lower()}_passed", has_passed)

        # Check if all tests have passed for the file type
        if self.file_type == FileType.SOURCE and self.import_passed and self.code_passed:
                self.set_all_status_passed()
        elif self.file_type == FileType.TEST and self.import_passed and self.pytest_passed:
                self.set_all_status_passed()
        return self
    
    def run_test(self, test_type: TestType, venv_name=str)-> Tuple[FileStatus, bool, str, str]:
        """
        Run a test based on the specified test type.

        Args:
            test_type (TestType): The type of test to run (IMPORT, CODE, or PYTEST).
            venv_name (str): The name of the virtual environment.

        Returns:
            Tuple[FileStatus, bool, str, str]: A tuple containing the file status, a boolean indicating whether the test passed or failed, and the error message (if any).
        """
        # Create a PythonExecutor instance with the provided virtual environment name
        from .virtual_env_manager import PythonExecutor
        run_python = PythonExecutor(venv_name)
        log(f"Running test {test_type.name} in {self.filename}", "INFO")
        if test_type == TestType.PYTEST:
            # Get the test filename
            test_filename = self.filename
            # Execute the code string and get the result
            result = run_python.run_pytest(test_filename)
            # get error details
            failed_tests = result['failed_tests']
            # format a error message
            error_message = "\n".join([
                f"{i+1}. {message['error_message']}" for i, message in enumerate(failed_tests)
            ])
            # get collection errors
            collection_errors = result['collection_errors']
            collection_errors = "\n".join([
                f"{i+1+len(failed_tests)}. {message['error_message']}" for i, message in enumerate(collection_errors)
            ])
            # combine the test and collection errors
            if error_message and collection_errors:
                error_message = error_message + "\n" + collection_errors
            elif error_message:
                pass
            elif collection_errors:
                error_message = collection_errors
            code_string = self.content.strip()
            # Check if the pytest passed
            has_passed = not bool(failed_tests) and not bool(collection_errors)
            if has_passed:
                return FileStatus.PYTEST_PASSED, has_passed, error_message, code_string
            else:
                return FileStatus.PYTEST_FAILED, has_passed, error_message, code_string
        elif test_type == TestType.IMPORT:
            # Get the import string from the solution dictionary
            code_string = self.solution.get('imports', '').strip()
        elif test_type == TestType.CODE:
            # Get the entire code string 
            code_string = self.content.strip()
        else:
            raise ValueError(f"Invalid test type specified: {test_type}")

        # Execute the code string and get the result
        result = run_python.exec_python_string(code_string)
        # Check if the code test passed
        has_passed = result.get('test_passed', False)
        # Get the error message from the result
        error_message = result.get('error_message', '')

        if has_passed:
            if test_type == TestType.IMPORT:
                return FileStatus.IMPORT_TEST_PASSED, has_passed, error_message, code_string
            elif test_type == TestType.CODE:
                return FileStatus.CODE_TEST_PASSED, has_passed, error_message, code_string
        else:
            if test_type == TestType.IMPORT:
                return FileStatus.IMPORT_TEST_FAILED, has_passed, error_message, code_string
            elif test_type == TestType.CODE:
                return FileStatus.CODE_TEST_FAILED, has_passed, error_message, code_string
    
    def set_status(self, status: FileStatus):
        self.status = status

    def all_tests_passed(self):
        return self.status == FileStatus.ALL_TESTS_PASSED
    
    @staticmethod
    def keep_last(managed_messages):
        seen = set()
        result = []
        last_role = None
        for role, message in reversed(managed_messages):
            if message not in seen:
                seen.add(message)
                if last_role is None or role != last_role:
                    result.append((role, message))
                    last_role = role
        return list(reversed(result))
    
    def manage_file_history(self, max_messages):
        if len(self.messages) <= max_messages:
            return # No need to manage if we have fewer messages than the max
        original_context = self.messages[0] if self.messages else None
        recent_messages = self.messages[-max_messages:]
        # preserve insertion order
        managed_messages = [msg for msg in [original_context] + recent_messages if msg]
        # remove duplicates and keep the last occurance if any (hard to keep user/role alternation this way)
        # self.messages = self.keep_last(managed_messages)
        self.messages = managed_messages

    def update_messages(self, new_message: Tuple[str, str], max_messages: int = 4):
        self.messages.append(new_message)
        self.manage_file_history(max_messages)

class SharedState(BaseModel):
    config_path: str = ''
    messages: list[tuple[str, str]] = Field(default_factory=list)
    baseline_file_names: dict[str, str]
    baseline_test_file_names: dict[str, str]
    source_file_states: dict[str, FileState] = Field(default_factory=dict)
    test_file_states: dict[str, FileState] = Field(default_factory=dict)
    source_file_status: bool = False
    test_file_status: bool = False
    venv_created: bool = False
    all_tests_passed: bool = False
    coverage: dict[str, Any] = Field(default_factory=dict)
    iteration_count: int = 0
    max_iterations: int = 10