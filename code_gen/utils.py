import os
import textwrap
import fnmatch
import json
import yaml
import re
from typing import Dict, List, Any, Optional
from langchain.prompts.chat import (
    ChatPromptTemplate, 
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from bs4 import BeautifulSoup as Soup
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from .logger import setup_logging, log
from pathlib import Path

setup_logging("INFO")

class TestResultSummary:
    """
    Class representing a summary of test results for a single file.
    """

    def __init__(self, max_message_length: int = 300):
        self.status = None
        self.iteration_count = None
        self.import_passed = False
        self.code_passed = False
        self.pytest_passed = False
        self.error = ''
        self.message = None
        self.max_message_length = max_message_length

    def __repr__(self):
        truncated_message = self.message[0], self.message[1][:self.max_message_length]
        return (
            f"Status: {self.status}\n"
            f"Iteration Count: {self.iteration_count}\n"
            f"Import Passed: {self.import_passed}\n"
            f"Code Passed: {self.code_passed}\n"
            f"Pytest Passed: {self.pytest_passed}\n"
            f"Error: {self.error}\n"
            #f"Message: {truncated_message}"
        )
    
class PrintTestResultSummary:
    """
    Class for generating and printing a summary of test results.

    Example usage:
    
    final_state = ... # Assuming you have a FinalState object
    print_summary = PrintTestResultSummary(final_state)
    print_summary.print_result_summary()
    """

    def __init__(self, final_state, max_message_length: int = 300):
        self.final_state = final_state
        self.max_message_length = max_message_length

    def generate_result_summary(self) -> Dict[str, Dict[str, TestResultSummary]]:
        """
        Generate a summary of test results for source code files and pytest files.

        Returns:
            dict: A dictionary containing the summary of test results for source code files and pytest files.
        """
        result_summary = {
            "Source": self._generate_file_summary(self.final_state.source_file_states),
            "Pytest": self._generate_file_summary(self.final_state.test_file_states),
        }

        return result_summary

    def _generate_file_summary(self, file_states) -> Dict[str, TestResultSummary]:
        """
        Generate a summary of test results for a set of files.

        Args:
            file_states (dict): A dictionary containing the file states for the files.

        Returns:
            dict: A dictionary containing the summary of test results for the files.
        """
        summary = {}

        for name, file_state in file_states.items():
            result_summary = TestResultSummary(self.max_message_length)
            result_summary.status = file_state.status.name
            result_summary.iteration_count = file_state.iteration_count
            result_summary.import_passed = file_state.import_passed
            result_summary.code_passed = file_state.code_passed
            result_summary.pytest_passed = file_state.pytest_passed
            role, message = file_state.messages[-1]
            result_summary.message = (role, message)
            result_summary.error = file_state.error
            summary[file_state.filename] = result_summary

        return summary

    def print_result_summary(self):
        """
        Print the summary of test results for source code files and pytest files.
        """
        print(f"All Test Passed: {self.final_state.all_tests_passed}")

        result_summary = self.generate_result_summary()

        # Print summary for source code files
        print("\nSource Code Files:")
        for filename, summary in result_summary["Source"].items():
            print(f"\n{filename}:")
            print(summary)

        # Print summary for pytest files
        print("\nPytest Files:")
        for filename, summary in result_summary["Pytest"].items():
            print(f"\n{filename}:")
            print(summary)

class PromptHandler:
    @staticmethod
    def load_prompts(file_path):
        """
        Loads the prompts from a YAML file.

        Args:
            file_path (str): The path to the YAML file containing the prompts.

        Returns:
            Dict: The loaded prompts.

        Raises:
            FileNotFoundError: If the specified file is not found.
            yaml.YAMLError: If there's an error parsing the YAML file.
        """
        return FileHandler.load_yaml_config(file_path)

    @staticmethod
    def get_prompt(file_path: str, prompt_name: str):
        """
        Retrieves a specific prompt from a YAML file.

        Args:
            file_path (str): The path to the YAML file containing the prompts.
            prompt_name (str): The name of the prompt to retrieve.

        Returns:
            Tuple[str, str]: A tuple containing the role and content of the prompt.

        Raises:
            ValueError: If the requested prompt is not found in the YAML file.
        """
        prompts = PromptHandler.load_prompts(file_path)
        if prompt_name in prompts:
            return prompts[prompt_name]['role'], prompts[prompt_name]['content']
        else:
            raise ValueError(f"Prompt '{prompt_name}' not found in the YAML file {file_path}.")

    @staticmethod
    def format_prompt(prompt, params):
        """
        Formats a prompt string with the given parameters.

        Args:
            prompt (str): The prompt string to format.
            params (Dict): A dictionary of parameters to use in formatting.

        Returns:
            str: The formatted prompt string.
        """
        return textwrap.dedent(prompt.format(**params))
    
    @staticmethod
    def get_required_params(prompt):
        """
        Extracts the required parameters from a prompt string.

        Args:
            prompt (str): The prompt string.

        Returns:
            List[str]: A list of required parameter names.
        """
        pattern = r'\{(\w+)\}'
        return re.findall(pattern, prompt)
    
    @staticmethod
    def check_prompt_params(prompt, params):
        """
        Checks if all required parameters are provided for a given prompt.

        Args:
            prompt (str): The prompt string.
            params (dict): A dictionary of parameters.

        Returns:
            bool: True if all required parameters are provided, False otherwise.
            set: A set of missing parameters, or an empty set if all required parameters are provided.
        """
        required_params = set(PromptHandler.get_required_params(prompt=prompt))
        provided_params = set(params)

        missing_params = required_params - provided_params
        if missing_params:
            return False, missing_params
        return True, set()

    @staticmethod
    def get_prompt_mapping(config: Dict[str, Any], prompt_type: str) -> Optional[Dict[str, str]]:
        """
        Retrieves the prompt mapping for a specific prompt type from the configuration.

        Args:
            config (Dict[str, Any]): The loaded configuration.
            prompt_type (str): The type of prompt mapping to retrieve.

        Returns:
            Optional[Dict[str, str]]: The prompt mapping for the specified type, or None if not found.
        """
        prompt_settings = config.get('prompt_settings', {})
        prompt_name_mapping = prompt_settings.get('prompt_name_mapping', {})
        return prompt_name_mapping.get(prompt_type)

    @staticmethod
    def resolve_file_path(base_path: str, relative_path: str) -> str:
        """
        Resolves a relative file path to an absolute path.

        Args:
            base_path (str): The base path to resolve from.
            relative_path (str): The relative path to resolve.

        Returns:
            str: The resolved absolute path.
        """
        return str(Path(base_path).parent.joinpath(relative_path).resolve())
    
class FileHandler:
    @staticmethod
    def resolve_path(filepath: str) -> Path:
        """
        Resolve the given filepath. If it doesn't exist, try to resolve it
        relative to the project root.
        """
        path = Path(filepath)
        if path.is_absolute() and path.exists():
            return path
        
        # Get the directory of the current script
        script_dir = Path(__file__).resolve().parent
        # Define the project root directory (adjust as needed)
        project_root = script_dir.parent
        
        project_path = project_root / path
        if project_path.exists():
            return project_path
        
        raise FileNotFoundError(f"Cannot find file: {filepath}")

    @staticmethod
    def load_yaml_config(file_path: str = 'breast_cancer.yaml'):
        try:
            config_path = FileHandler.resolve_path(file_path)
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            return config
        except FileNotFoundError as e:
            print(f"Error loading configuration: {e}")
            return {}
        except yaml.YAMLError as e:
            print(f"Error parsing YAML: {e}")
            return {}
        
    @staticmethod
    def remove_leading_whitespace(code):
        return textwrap.dedent(code).strip()
        
    @staticmethod
    def read_file(filename: str):
        content = ""
        try:
            with open(filename, "r") as f:
                content = f.read()
        except FileNotFoundError:
            log(f"{filename} file not found.", "ERROR")
        return content

    @staticmethod
    def get_pytest_ini(venv_name: str = '.venv', config_path: str = '') -> str:
        config = FileHandler.load_yaml_config(config_path)
        pytest_ini_content = config['pytest_config']['content']
        return pytest_ini_content.format(venv_name=venv_name)

    @staticmethod
    def create_pytest_ini(dir_name: str = '', venv_name: str = '.venv', config_path: str = ''):
        pytest_ini_content = FileHandler.get_pytest_ini(venv_name, config_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name,'pytest.ini'), 'w') as f:
            f.write(pytest_ini_content)
        log("Created pytest.ini", "INFO")
    
    @staticmethod
    def create_ini_file(dir_name: str = ''):
        with open(os.path.join(dir_name,'__init__.py'), 'w') as f:
            f.write('')
        log(f"Created __init__.py in {dir_name}", "INFO")        

    @staticmethod
    def get_conftest_py(venv_name: str = '.venv', config_path: str = '') -> str:
        config = FileHandler.load_yaml_config(config_path)
        conftest_content = config['conftest']['content']
        return conftest_content.format(venv_name=venv_name)

    @staticmethod
    def create_conftest_py(dir_name: str = '', venv_name: str = '.venv', config_path: str = ''):
        conftest_content = FileHandler.get_conftest_py(venv_name, config_path)
        test_dir = ''
        if 'tests' not in dir_name:
            test_dir = os.path.join(dir_name, 'tests')
        os.makedirs(test_dir, exist_ok=True)
        with open(os.path.join(test_dir, 'conftest.py'), 'w') as f:
            f.write(conftest_content)
        log("Created tests/conftest.py", "INFO")

    @staticmethod
    def get_coveragerc(config_path: str = '') -> str:
        config = FileHandler.load_yaml_config(config_path)
        coveragerc_content = config['coveragerc']['content']
        return coveragerc_content

    @staticmethod
    def create_coveragerc(dir_name: str = '', config_path: str = ''):
        coveragerc_content = FileHandler.get_coveragerc(config_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(os.path.join(dir_name,'.coveragerc'), 'w') as f:
            f.write(coveragerc_content)
        log("Created .coveragerc", "INFO")

    @staticmethod
    def create_file(file_data):
        """
        Creates a file with the specified content and file path.

        Args:
            file_data (dict): A dictionary containing the following keys:
                - "content": The content to be written to the file.
                - "filename": The file path where the file should be created.
                - "file_type": The type of the file (e.g., "python" or "text").

        Raises:
            ValueError: If the file path is not provided.
        """
        content = file_data.get("content", "")

        file_path = file_data.get("filename", "")
        file_type = file_data.get("file_type", "")

        if file_path:
            directory_name = os.path.dirname(file_path)
            if directory_name and not os.path.exists(directory_name):
                os.makedirs(directory_name)

            file_extension = ".py" if file_type == "python" else ".txt"
            with open(f"{file_path.replace(file_extension, '')}{file_extension}", "w") as file:
                file.write(content)

            log(f"{file_type.capitalize()} file '{file_path}' has been created.", "INFO")
        else:
            raise ValueError("File path is not provided.")
    
    @staticmethod
    def parse_json(json_string):
        """
        Parses a JSON string and returns a Python dictionary.

        Args:
            json_string (str): The JSON string to be parsed.

        Returns:
            dict: A Python dictionary representing the parsed JSON data.

        Raises:
            ValueError: If the input string is not a valid JSON format.
        """
        try:
            json_string = json_string.strip()
            # generic approach
            start = json_string.find("{")
            end = json_string.rfind("}", start) + 1
            json_string = json_string[start:end]
            return json.loads(json_string)
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON response: {json_string}")
        
    @staticmethod
    def get_file_tree(directory, level=0, indent=4, exclude_dirs=None, include_files=None, unique_file_paths=None):
        """
        Recursively generates a file tree structure for the given directory.

        Args:
            directory (str): The path to the directory.
            level (int, optional): The current level of the directory tree. Defaults to 0.
            indent (int, optional): The number of spaces to use for indentation. Defaults to 4.
            exclude_dirs (list, optional): A list of directory names (or patterns) to exclude from the file tree.
            include_files (list, optional): A list of file names (or patterns) to include in the file tree.
            unique_file_paths (set, optional): A set to keep track of unique file paths.

        Returns:
            str: The file tree structure as a string.
        """
        tree = ""
        prefix = " " * level * indent
        # Initialize unique_file_paths if not provided
        if unique_file_paths is None:
            unique_file_paths = set()

        # Initialize exclude lists if not provided
        exclude_dirs = exclude_dirs or []
        include_files = include_files or []

        for root, dirs, files in os.walk(directory):
            relative_root = os.path.relpath(root, directory)
            if relative_root == ".":
                relative_root = ""
            else:
                relative_root = os.path.join(relative_root)

            # Filter out excluded directories
            dirs[:] = [d for d in dirs if not any(fnmatch.fnmatch(d, pattern) for pattern in exclude_dirs)]

            # Draw directories
            for dir_name in dirs:
                dir_path = os.path.join(relative_root, dir_name)
                tree += f"{prefix}{'├' if files or dirs[dirs.index(dir_name)+1:] else '└'}── {dir_path}\n"
                tree += FileHandler.get_file_tree(os.path.join(directory, dir_path), level + 1, indent, exclude_dirs, include_files, unique_file_paths)

            # Draw files (only if not recursing into a directory)
            files = [f for f in files if any(fnmatch.fnmatch(f, pattern) for pattern in include_files)]
            for file_name in files:
                file_path = os.path.join(root, file_name)
                if file_path not in unique_file_paths:
                    unique_file_paths.add(file_path)
                    relative_file_path = os.path.relpath(file_path, directory)
                    tree += f"{prefix}{'└' if not dirs else '├'}── {relative_file_path}\n"

        return tree

class ChatPromptTemplateBuilder:
    @staticmethod
    def get_chat_templates(messages: List[Dict[str,Any]])->ChatPromptTemplate:
        """
        Construct a ChatPromptTemplate from a list of messages.

        Args:
            messages (List[Dict[str,Any]]): A list of messages, where each message is a dictionary with 'role' and 'content' keys.
        Returns:
            ChatPromptTemplate: The constructed ChatPromptTemplate.
        """
        chat_prompts = []
        for msg in messages:
            role = msg.get('role', None)
            if role is None:
                raise ValueError(f"Invalid role: {msg['role']}")
            if role in ['system']:
                template = SystemMessagePromptTemplate.from_template(msg['content'])
            elif role in ['user', 'placeholder']:
                template = HumanMessagePromptTemplate.from_template(msg['content'])
            else:
                raise ValueError(f"Invalid role: {msg['role']}")
            chat_prompts.append(template)
        return ChatPromptTemplate.from_messages(chat_prompts)
class UrlPageLoader:
    @staticmethod
    def concatenate_docs(docs: List[Any])-> str: 
        return "\n\n\n --- \n\n\n".join([doc.page_content for doc in docs])
    @staticmethod
    def sort_docs(docs):
        # Sort the list based on the URLs and get the text
        d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
        return list(reversed(d_sorted))
    @staticmethod
    def load_html_page(url: str, max_depth=20):
        loader = RecursiveUrlLoader(
            url=url, 
            max_depth=20, 
            extractor=lambda x: Soup(x, "html.parser").text
        )
        docs = loader.load()
        d_sorted = UrlPageLoader.sort_docs(docs)
        concatenated_content = UrlPageLoader.concatenate_docs(d_sorted)
        return concatenated_content
    @staticmethod
    def load_multiple_pages(urls: List[str], max_depth=20) -> str:
        docs = ''
        if len(urls) > 1:
            for url in urls:
                doc = UrlPageLoader.load_html_page(url, max_depth)
                docs += "\n\n\n --- \n\n\n" + doc
        else:
            for url in urls:
                doc = UrlPageLoader.load_html_page(url, max_depth)
                docs += doc
        return docs    