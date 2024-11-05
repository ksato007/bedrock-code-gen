from .graph_state import SharedState, TestType, FileState, FileType
from .config_manager import ConfigManager
from .utils import FileHandler, PromptHandler
import os
from typing import Dict, Tuple, Any
from pathlib import Path
from .logger import setup_logging, log
from .graph_state import FileStatus
import concurrent.futures

setup_logging("INFO")

class TestRunner:
    """
    The TestRunner class is responsible for running tests on source and test files, managing their states, 
    and handling test failures by generating prompts to fix import and code issues. 
    It initializes the necessary configurations, sets up a virtual environment, 
    and provides methods to add source and test file states, run import and code tests, 
    prepare prompts for fixing import and code issues, and retrieve source and test file names. 
    The class also utilizes a thread pool for parallel execution and provides methods for managing the thread pool's lifecycle.
    """
    def __init__(self, state: SharedState):
        """
        Initialize the TestRunner instance.

        Args:
            state (SharedState): The shared state object containing the project's state.
        
        Raises:
            RuntimeError: If the virtual environment directory does not exist or is not a directory.
        """
        configs = ConfigManager.from_yaml_file(state.config_path)
        self.state = state
        self.env_settings = configs.env_settings
        self.prompt_settings = configs.prompt_settings
        self.max_num_messages = self.prompt_settings.max_num_messages
        self.test_runner_settings = configs.test_runner_settings
        self.venv_name = self.env_settings.get('venv_name', 'venv')
        self.source_file_states = {}
        self.test_file_states = {}
        self.thread_pool = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def __del__(self):
        self.shutdown_thread_pool()

    def shutdown_thread_pool(self):
        """
        Shut down the thread pool executor.
        """
        if self.thread_pool:
            self.thread_pool.shutdown()

    def check_venv_setup(self) -> None:
        """
        Checks if the virtual environment is set up correctly.

        Returns:
            bool: True if the virtual environment is set up correctly, False otherwise.
        """
        log("Checking virtual environment setup", "INFO")
        cwd = Path.cwd()
        venv_dir = cwd / self.venv_name

        if venv_dir.exists() and venv_dir.is_dir():
            return
        else:
            error_message = f"Virtual environment directory '{venv_dir}' does not exist or is not a directory."
            log(error_message, "ERROR")
            raise RuntimeError(error_message)
    
    def get_source_file_names(self) -> list[str]:
        """
        Get a list of source file names.

        Returns:
            list[str]: A list of source file names.
        """
        return list(self.source_file_states.keys())
    
    def get_test_file_names(self) -> list[str]:
        """
        Get a list of test file names.

        Returns:
            list[str]: A list of test file names.
        """
        return list(self.test_file_states.keys())   
    
    def add_source_file_state(self, name: str, file_state: FileState):
        """
        Add a source file state to the TestRunner.

        Args:
            name (str): The name of the source file.
            file_state (FileState): The FileState object representing the source file.
        """
        self.source_file_states[name] = file_state
    
    def add_test_file_state(self, name: str, file_state: FileState):
        """
        Add a test file state to the TestRunner.

        Args:
            name (str): The name of the test file.
            file_state (FileState): The FileState object representing the test file.
        """
        self.test_file_states[name] = file_state
    
    def prepare_fix_import_prompt(self, file_state: FileState) -> FileState:
        # Get the prompt name for import fix from the prompt settings
        prompt_name = self.prompt_settings.prompt_name_mapping[self.test_runner_settings.fix_import_key]
        # Get the path to the prompt file
        prompt_filename = self.prompt_settings.template_file_path
        # Get the prompt and role from the PromptHandler
        role, prompt = PromptHandler.get_prompt(prompt_filename, prompt_name)
        # Generate import test prompt parameters dynamically
        params = {
            "filename": file_state.filename,
            "code": file_state.solution['code'],
            "error_message": file_state.error,
            "file_tree": FileHandler.get_file_tree(
                directory=os.getcwd(),
                exclude_dirs=self.test_runner_settings.directory_tree_dir_exclusion_list,
                include_files=self.test_runner_settings.directory_tree_file_inclusion_list
            )
        }
        # Check if required parameters are provided
        has_required_params, missing_params = PromptHandler.check_prompt_params(prompt, params)
        if not has_required_params:
            raise ValueError(f"Missing required parameters for prompt: {', '.join(missing_params)}")
        
        # Format the prompt with the provided parameters
        formatted_prompt = PromptHandler.format_prompt(prompt, params)
        # Append the formatted prompt to the messages list
        new_message = (role, formatted_prompt)
        file_state.update_messages(new_message, self.max_num_messages)
        # file_state.messages.append((role, formatted_prompt))
        return file_state
    
    def prepare_fix_code_prompt(self, file_state: FileState) -> FileState:
        prompt_name = self.prompt_settings.prompt_name_mapping[self.test_runner_settings.fix_code_key]
        prompt_filename = self.prompt_settings.template_file_path
        # Get the prompt and role from the PromptHandler
        role, prompt = PromptHandler.get_prompt(prompt_filename, prompt_name)
        # Generate prompt parameters dynamically
        params = {
            "filename": file_state.filename,
            "code": file_state.solution['code'],
            "error_message": file_state.error,
        }
        # Check if required parameters are provided
        has_required_params, missing_params = PromptHandler.check_prompt_params(prompt, params)
        if not has_required_params:
            raise ValueError(f"Missing required parameters for prompt: {', '.join(missing_params)}")
        
        # Format the prompt with the provided parameters
        formatted_prompt = PromptHandler.format_prompt(prompt, params)
        # Append the formatted prompt to the messages list
        new_message = (role, formatted_prompt)
        file_state.update_messages(new_message, self.max_num_messages)
        # file_state.messages.append((role, formatted_prompt))
        return file_state
            
    def run_import_test(self, file_state: FileState):
        """
        Run import tests on a file.

        This method runs import tests on the provided `FileState` object. It retrieves the prompt name
        for import fix from the `test_runner_settings`, and the prompt file path from the `prompt_settings`.
        It then calls the `handle_test` method of the `FileState` object with the appropriate parameters
        to run the import test.

        Args:
            file_state (FileState): The FileState object representing the file to be tested.
            prompt_params (Optional[Dict[str, str]]): An optional dictionary of prompt parameters
                to be passed to the `handle_test` method. If not provided, an empty dictionary is used.

        Returns:
            FileState: The updated FileState object after running the import tests.
        """
        # Run the import test on the file
        new_file_state = file_state.handle_test(
            test_type=TestType.IMPORT,
            venv_name=self.venv_name
        )
        # Handle Result
        if new_file_state.import_passed:
            return new_file_state
        # If import failed, insert message to fix import issues
        new_file_state = self.prepare_fix_import_prompt(new_file_state)          
        return new_file_state
    
    def run_code_test(self, file_state: FileState):
        """
        Run code tests on a file.

        This method runs code tests on the provided `FileState` object. It retrieves the prompt name
        for code fix from the `test_runner_settings`, and the prompt file path from the `prompt_settings`.
        It then calls the `handle_test` method of the `FileState` object with the appropriate parameters
        to run the code test.

        Args:
            file_state (FileState): The FileState object representing the file to be tested.
            prompt_params (Optional[Dict[str, str]]): An optional dictionary of prompt parameters
                to be passed to the `handle_test` method. If not provided, an empty dictionary is used.

        Returns:
            FileState: The updated FileState object after running the code tests.
        """
        new_file_state = file_state.handle_test(
            test_type=TestType.CODE,
            venv_name=self.venv_name
        )
        # Handle Result
        if new_file_state.code_passed:
            return new_file_state
        # If code test failed, insert message to fix import issues
        new_file_state = self.prepare_fix_code_prompt(new_file_state)
        return new_file_state

    def get_pytest_prompt_settings(self) -> Dict[str, str]:
        fix_pytest_key = self.prompt_settings.prompt_name_mapping[self.test_runner_settings.fix_pytest_key]
        prompt_filename = self.prompt_settings.template_file_path
        role, fix_pytest_template = PromptHandler.get_prompt(prompt_filename, fix_pytest_key)
        test_instructions_key = 'test_instructions'  # Assuming this is the key for test instructions in your template
        _, test_instructions = PromptHandler.get_prompt(prompt_filename, test_instructions_key)
        return {
            'role': role,
            'fix_pytest_template': fix_pytest_template,
            'test_instructions': test_instructions
        }
    
    def get_common_prompt_params(self, source_file_state: FileState, test_file_state: FileState) -> Dict[str, Any]:
        return {
            "source_filename": source_file_state.filename,
            "source_code": source_file_state.solution,
            "test_filename": test_file_state.filename,
            "test_code": test_file_state.solution,
            "error_message": test_file_state.error,
            "iteration_count": self.state.iteration_count
        }

    def get_file_specific_params(self, common_params: Dict[str, Any], file_state: FileState, file_type: str, is_test_file: bool) -> Dict[str, Any]:
        params = common_params.copy()
        params.update({
            "filename": file_state.filename,
            "test_instructions": self.get_pytest_prompt_settings()['test_instructions'] if is_test_file else ""
        })
        return params

    def validate_prompt_params(self, prompt: str, params: Dict[str, Any], param_type: str):
        has_required_params, missing_params = PromptHandler.check_prompt_params(prompt, params)
        if not has_required_params:
            raise ValueError(f"Missing required parameters for {param_type} prompt: {', '.join(missing_params)}")

    def format_pytest_prompt(self, prompt: str, params: Dict[str, Any]) -> str:
        return PromptHandler.format_prompt(prompt, params)

    def update_file_states(self, 
                           source_file_state: FileState, 
                           test_file_state: FileState, 
                           source_formatted_prompt: str, 
                           test_formatted_prompt: str, 
                           role: str) -> Tuple[FileState, FileState]:
        source_file_state.error = test_file_state.error
        new_source_message = (role, source_formatted_prompt)
        new_test_message = (role, test_formatted_prompt)
        source_file_state.update_messages(new_source_message, self.max_num_messages)
        test_file_state.update_messages(new_test_message, self.max_num_messages)
        return source_file_state, test_file_state

    def prepare_pytest_prompt(self, source_file_state: FileState, test_file_state: FileState) -> Tuple[FileState, FileState]:
        """
        Prepare the prompt for fixing pytest issues.

        Args:
            source_file_state (FileState): The file state object for the source file.
            test_file_state (FileState): The file state object for the test file.

        Returns:
            Tuple[FileState, FileState]: The updated source and test file state objects with the formatted prompts appended.
        """
        log(f"Preparing pytest prompt for {source_file_state.filename} and {test_file_state.filename}", "INFO")

        try:
            prompt_settings = self.get_pytest_prompt_settings()


            common_params = self.get_common_prompt_params(source_file_state, test_file_state)
            
            source_params = self.get_file_specific_params(common_params, source_file_state, "source", is_test_file=False)
            test_params = self.get_file_specific_params(common_params, test_file_state, "test", is_test_file=True)

            self.validate_prompt_params(prompt_settings['fix_pytest_template'], source_params, "source")
            self.validate_prompt_params(prompt_settings['fix_pytest_template'], test_params, "test")

            source_formatted_prompt = self.format_pytest_prompt(prompt_settings['fix_pytest_template'], source_params)
            test_formatted_prompt = self.format_pytest_prompt(prompt_settings['fix_pytest_template'], test_params)
            source_file_state, test_file_state = self.update_file_states(
                source_file_state, test_file_state, 
                source_formatted_prompt, test_formatted_prompt, 
                prompt_settings['role']
            )

            return source_file_state, test_file_state

        except Exception as e:
            log(f"Error preparing pytest prompt: {str(e)}", "ERROR")
            raise

    def run_pytest(self, file_state: FileState):
        """
        Run pytest tests on a file.

        This method runs pytest tests on the provided `FileState` object. It retrieves the prompt name
        for pytest fix from the `test_runner_settings`, and the prompt file path from the `prompt_settings`.
        It then calls the `handle_test` method of the `FileState` object with the appropriate parameters
        to run the pytest test.

        Args:
            file_state (FileState): The FileState object representing the file to be tested.
            prompt_params (Optional[Dict[str, str]]): An optional dictionary of prompt parameters
                to be passed to the `handle_test` method. If not provided, an empty dictionary is used.

        Returns:
            FileState: The updated FileState object after running the pytest tests.
        """
        new_file_state = file_state.handle_test(
            test_type=TestType.PYTEST,
            venv_name=self.venv_name
        )
        # Handle Result
        if new_file_state.pytest_passed:
            return new_file_state
        # in case of pytest error we want to analyze & fix source and test separately
        name = os.path.basename(new_file_state.filename).split(".")[0].replace("test_", "")
        new_source_state, new_file_state = self.prepare_pytest_prompt(source_file_state=self.state.source_file_states[name], test_file_state=new_file_state)

        # update source file state in place
        new_source_state.status = FileStatus.PYTEST_FAILED
        self.state.source_file_states[name] = new_source_state
        return new_file_state

    def run_tests_on_file(self, file_state: FileState) -> FileState:
        """
        Run appropriate tests on a file based on its state and type.

        Args:
            file_state (FileState): The FileState object representing the file.

        Returns:
            FileState: The updated FileState object after running the tests.
        """
        file_type = file_state.file_type

        # If all tests have already passed, return the file state as is
        if file_state.status == FileStatus.ALL_TESTS_PASSED:
            return file_state
        
        # Run import tests if they haven't passed yet
        elif not file_state.import_passed:
            file_state = self.run_import_test(file_state)
        # If import test passed, run code tests for source files or pytest tests for test files
        is_run_code_test = file_state.import_passed and not file_state.code_passed and file_type == FileType.SOURCE
        if is_run_code_test:
            file_state = self.run_code_test(file_state)
        # If import test passed, run pytest tests if they haven't passed yet and the file is a test file
        is_run_pytest =  file_state.import_passed and not file_state.pytest_passed and file_type == FileType.TEST
        if is_run_pytest:
            file_state = self.run_pytest(file_state)

        return file_state
    @staticmethod
    def is_not_pending(status: FileStatus) -> bool:
        """
        Check if the given FileStatus is not in a pending state.

        Args:
            status (FileStatus): The FileStatus object to check.

        Returns:
            bool: True if the status is not pending, False otherwise.
        """
        pending_statuses = [
            FileStatus.IMPORT_TEST_PENDING,
            FileStatus.CODE_TEST_PENDING,
            FileStatus.PYTEST_PENDING,
        ]
        return status not in pending_statuses

    def run_tests_on_all_test_files_serial(self):
        """
        Run tests on all test files in a serial manner.

        Returns:
            SharedState: The updated shared state object after running the tests.
        """
        for name, file_state in self.test_file_states.items():
            if file_state.status == FileStatus.ALL_TESTS_PASSED or TestRunner.is_not_pending(file_state.status):
                continue
            new_state = self.run_tests_on_file(file_state)
            self.state.test_file_states[name] = new_state
        return self.state

    def run_tests_on_all_test_files_threaded(self):
        """
        Run tests on all test files in a serial manner.

        Returns:
            SharedState: The updated shared state object after running the tests.
        """
        log("Running tests on all test files (threaded)")

        with self.thread_pool as executor:
            futures = []
            for name, file_state in self.test_file_states.items():
                if file_state.status == FileStatus.ALL_TESTS_PASSED or TestRunner.is_not_pending(file_state.status):
                    continue
                future = executor.submit(self.run_tests_on_file, file_state)
                futures.append((name, future))
            for name, future in futures:
                new_state = future.result()
                self.state.test_file_states[name] = new_state
        return self.state

    def run_tests_on_all_source_files_serial(self):
        """
        Run tests on all source files in a serial manner.

        Returns:
            SharedState: The updated shared state object after running the tests.
        """
        for name, file_state in self.source_file_states.items():
            if file_state.status == FileStatus.ALL_TESTS_PASSED or TestRunner.is_not_pending(file_state.status):
                continue
            new_state = self.run_tests_on_file(file_state)
            self.state.source_file_states[name] = new_state
        return self.state
    
    def run_tests_on_all_source_files_threaded(self):
        """
        Run tests on all source files in a serial manner.

        Returns:
            SharedState: The updated shared state object after running the tests.
        """
        log("Running tests on all source files (threaded)")
        with self.thread_pool as executor:
            futures = []
            for name, file_state in self.source_file_states.items():
                if file_state.status == FileStatus.ALL_TESTS_PASSED or TestRunner.is_not_pending(file_state.status):
                    continue
                future = executor.submit(self.run_tests_on_file, file_state)
                futures.append((name, future))
            for name, future in futures:
                new_state = future.result()
                self.state.source_file_states[name] = new_state
        return self.state
    
    @staticmethod
    def check_status(file_states: Dict[str, FileState], excluded_files = []) -> bool:
        all_features_passed = True
        for name, file_state in file_states.items():
            if name in excluded_files:
                continue
            is_passed = file_state.status == FileStatus.ALL_TESTS_PASSED
            all_features_passed &= is_passed
        return all_features_passed

    def check_status_source_files(self) -> bool:
        excluded_files = self.test_runner_settings.primary_test_exclusion_files
        return self.check_status(self.state.source_file_states, excluded_files=excluded_files)

    def check_status_test_files(self) -> bool:
        excluded_files = []
        return self.check_status(self.state.test_file_states, excluded_files=excluded_files)

    def check_coverage(self, file_path: str = 'tests/'):
        from .virtual_env_manager import PythonExecutor
        run_python = PythonExecutor(self.venv_name)
        self.state.coverage = run_python.run_coverage(file_path=file_path)
        return self.state
            