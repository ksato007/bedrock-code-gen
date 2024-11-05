from typing import List, Dict, Any, Union, Optional, Type, Callable
from enum import Enum
import xml.etree.ElementTree as ET
from pydantic import ValidationError
import concurrent.futures
from threading import Lock
from pathlib import Path
import os

# langchain
from langchain_core.pydantic_v1 import BaseModel

# custom modules
from .model_manager import ModelManager
from .ai_message_parser import AIMessageParser
from .code_models import Code, create_baseline_code_model
from .logger import setup_logging, log
from .utils import FileHandler, PromptHandler
from .graph_state import FileState, SharedState, FileStatus, FileType
from .logger import log
from .config_manager import ConfigManager
from .test_runner import TestRunner
import time
import concurrent


class ToolClasses:
    """
    A class that holds the available tool classes for the model.
    """
    CODE: Type[Code] = Code # type: ignore

class SchemaType:
    """
    A class that holds the available schema types for parsing the model output.
    """
    CODE: Type[Code] = Code  # type: ignore

class CodeType(Enum):
    """
    An enumeration representing the different types of code that can be generated.
    """
    SOURCE = 'SOURCE'
    TEST = 'TEST'
    IMPORTS = 'IMPORTS'

class CodeGenerator:
    """
    The CodeGenerator class is responsible for generating code using a language model and managing the state of the code generation process: 
        1. initializes the necessary configurations
        2. sets up the environment and prompt settings
        3. loads the required tools and baselines
        4. creates a thread pool for parallel execution.
        5. set up the chat model
        6. generating code with retries
        7. run test cases on the generated code.  
    """
    def __init__(self, state: SharedState) -> SharedState:
        """

        Initialize the CodeGenerator instance.

        Args:
            state (SharedState): The shared state object containing the project's state.
            model_id (str): The ID of the model to be used.

        Returns:
            SharedState: The updated shared state.
        """
        self.state = state
        self.configs = ConfigManager.from_yaml_file(self.state.config_path)
        self.model_settings = self.configs.bedrock_model_settings
        self.setup_env_settings()
        self.setup_prompt_settings()
        self.setup_tools()
        self.setup_chat_model_settings()
        self.setup_baselines()
        self.thread_pool = None
        self.thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=5)

    def setup_env_settings(self):
        self.python_version = self.configs.env_settings['python_version']

    def setup_prompt_settings(self):
        self.prompt_settings = self.configs.prompt_settings
        self.max_num_messages = self.prompt_settings.max_num_messages
        self.test_runner_settings = self.configs.test_runner_settings
        
    def setup_chat_model_settings(self):
        self.model_id = self.model_settings.model_id or ''
        self.model_kwargs = self.model_settings.inference_modifier or {}
        self.model_max_retry = self.model_settings.max_retries
        self.model_retry_delay = self.model_settings.retry_delay
        self.model = self.get_chat_model()

    def setup_tools(self):
        self.tools: List[Type[Code]] = []
        for tool_name in self.model_settings.tools or []:
            tool_class = getattr(ToolClasses, tool_name.upper(), None)
            if tool_class is not None:
                self.tools.append(tool_class)
            else:
                log(f"Warning: Tool '{tool_name}' not found in ToolClasses", "WARNING")

    def setup_baselines(self):
        self.baseline_code = create_baseline_code_model(self.state.baseline_file_names)
        self.test_baseline_code = create_baseline_code_model(self.state.baseline_test_file_names)
        SchemaType.BASELINE = self.baseline_code
        SchemaType.TESTCODE = self.test_baseline_code

    def __del__(self):
        self.shutdown_thread_pool()

    def shutdown_thread_pool(self):
        """
        Shut down the thread pool executor.
        """
        if self.thread_pool:
            self.thread_pool.shutdown()
            
    def initialize_test_runner(self):
        """
        Initialize the TestRunner instance with the source and test file states.

        Returns:
            TestRunner: The initialized TestRunner instance.
        """
        log("Initializing TestRunner", "INFO")
        test_runner = TestRunner(self.state)
        for name, file_state in self.state.source_file_states.items():
            if name not in self.test_runner_settings.primary_test_exclusion_files:
                # log(f"Adding source file: {name}. exlusion_list={self.test_runner_settings.primary_test_exclusion_files}", "INFO")
                test_runner.add_source_file_state(name=name, file_state=file_state)
        for name, file_state in self.state.test_file_states.items():
            if name not in self.test_runner_settings.primary_test_exclusion_files:
                test_runner.add_test_file_state(name=name, file_state=file_state)
        return test_runner    

    def get_chat_model(self):
        """
        Get the chat model instance based on the configuration settings.

        Returns:
            Any: The chat model instance.
        """
        model_manager = ModelManager(
            model_id=self.model_id, 
            model_kwargs=self.model_kwargs, 
            tools=self.tools
        )
        return model_manager.create_model()

    def generate_code(self, schema: Optional[Type[BaseModel]], messages: list) -> Union[BaseModel, str]:
        """
        Generate code based on the provided messages and optional schema.

        Args:
            messages (list): A list of messages to be passed to the model.
            schema (Optional[Type[BaseModel]]): The schema to use for parsing the output. Defaults to None.

        Returns:
            Union[BaseModel, str]: If a valid schema is provided, returns the parsed model output.
                Otherwise, returns the raw model output as a string.
        """
        log("Invoking model with the prompt", "INFO")
        output = self.model.invoke(messages)
        if schema is not None and issubclass(schema, BaseModel):
            try:
                return AIMessageParser.parse_ai_message(output, schema)
            except Exception as e:
                log(f"Error parsing output with schema: {e}", "ERROR")
                log(f"Output: {output}", "INFO")
                raise e
        else:
            return output
    
    def generate_code_with_retry(self, schema: Optional[Type[BaseModel]], messages: list) -> Union[BaseModel, str]:
        """
        Generate code with retry based on the provided messages and optional schema.

        Args:
            messages (list): A list of messages to be passed to the model.
            schema (Optional[Type[BaseModel]]): The schema to use for parsing the output. Defaults to None.

        Returns:
            Union[BaseModel, str]: If a valid schema is provided, returns the parsed model output.
                Otherwise, returns the raw model output as a string.
        """
        num_retries = 0
        last_exception = None

        while (num_retries < self.model_max_retry):
            try:
                output = self.model.invoke(messages)
                if schema is not None and issubclass(schema, BaseModel):
                    try:
                        return AIMessageParser.parse_ai_message(output, schema)
                    except (ET.ParseError, ValueError) as e:
                        last_exception = e
                        log(f"Error parsing output with schema: {e}", "ERROR")
                        log(f"Output: {output}", "INFO")
                        # log(f"ParseError: {e}", "ERROR")
                        num_retries += 1
                        if num_retries == self.model_max_retry:
                            log("Maximum number of retries reached. Raising exception.", "ERROR")
                            raise e
                        else:
                            log(f"Retrying in {self.model_retry_delay} seconds...", "WARNING")
                            time.sleep(self.model_retry_delay)
                    except Exception as e:
                        last_exception = e
                        log(f"Error parsing output with schema: {e}", "ERROR")
                        log(f"Output: {output}", "INFO")
                        raise e
                else:
                    return output
            except Exception as e:
                last_exception = e
                log(f"Error invoking model: {e}", "ERROR")
                raise e
        # If we reach this point, it means the maximum number of retries has been exceeded
        log(f"Maximum number of retries exceeded. Last exception: {last_exception}", "ERROR")
        return output

    def load_codebase_config(self):
        config_file_path = self.prompt_settings.config_file_path
        return FileHandler.load_yaml_config(config_file_path)
    
    def get_file_config(self, codebase_config: Dict[str, Any], file_alias: str, default_file_name: str) -> Dict[str, Any]:
        name = Path(file_alias).stem # w/o .py
        file_config = codebase_config.get('modules', {}).get(name, {})
        return {
            'description': file_config.get('description', ''),
            'steps': file_config.get('steps', ''),
            'imports': file_config.get('imports', ''),
            'filename': file_config.get('filename', default_file_name)
        }
    
    def format_project_context(self, prompt_settings: Dict[str, Any], codebase_config: Dict[str, Any]) -> str:
        return prompt_settings['project_context_template'].format(
            project_name=codebase_config['project_name'],
            dataset_name=codebase_config['dataset_names'],
            project_structure=codebase_config['project_structure'],
            python_version=self.python_version
        )
    
    def format_code_template(self, prompt_settings: Dict[str, Any], file_config: Dict[str, Any]) -> str:
        return prompt_settings['module_code_format_template'].format(
            description=file_config['description'].strip(),
            steps=file_config['steps'].strip(),
            imports=file_config['imports'].strip(),
            filename=file_config['filename'].strip()
        )

    def get_codebase_settings(self, codebase_type: CodeType):
        if codebase_type == CodeType.SOURCE:
            return self.state.baseline_file_names, self.baseline_code, 'source_file_states'
        elif codebase_type == CodeType.TEST:
            return self.state.baseline_test_file_names, self.test_baseline_code, 'test_file_states'
        else:
            raise ValueError(f"Invalid codebase type: {codebase_type}")
        
    def format_module_prompt(self, prompt_settings: Dict[str, Any], project_context: str, filename: str, code_format: str) -> str:
        return prompt_settings['module_prompt_template'].format(
            project_context=project_context,
            filename=filename,
            module_code_format=code_format
        )
    
    def get_prompt_settings(self):
        template_file_path = self.prompt_settings.template_file_path
        # get generic template names
        project_context_template_name = self.prompt_settings.prompt_name_mapping['generate_code']['context']
        module_prompt_template_name = self.prompt_settings.prompt_name_mapping['generate_code']['module']
        module_code_format_template_name = self.prompt_settings.prompt_name_mapping['generate_code']['code_format']
        # load the generic templates
        _, project_context_template = PromptHandler.get_prompt(template_file_path, project_context_template_name)
        role, module_prompt_template = PromptHandler.get_prompt(template_file_path, module_prompt_template_name)
        _, module_code_format_template = PromptHandler.get_prompt(template_file_path, module_code_format_template_name)
        return {
            'role': role, # 'user'
            'project_context_template': project_context_template,
            'module_prompt_template': module_prompt_template,
            'module_code_format_template': module_code_format_template,
            'filename': template_file_path
        }
    def generate_baseline_code_dict(self, file_names: Dict[str, str], generate_func: Callable) -> Dict[str, Any]:
        baseline_code_dict = {}
        message_dict = {}
        with self.thread_pool as executor:
            future_to_file = {
                executor.submit(generate_func, file_name, file_alias): (file_name, file_alias) 
                for file_name, file_alias in file_names.items()
            }
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    file_alias, code, messages = future.result()
                    baseline_code_dict[file_alias] = code
                    message_dict[file_alias] = messages
                except Exception as e:
                    log(f"Error in thread for {future}: {str(e)}", "ERROR")
        return baseline_code_dict, message_dict
    
    def generate_codebase_multiple_prompts_threaded(self, codebase_type: CodeType) -> SharedState:
        """
        Generate the codebase for the specified type (baseline or test).

        Args:
            codebase_type (CodeType): The type of codebase to generate (SOURCE or TEST).

        Returns:
            SharedState: The updated shared state with the generated codebase.
        """
        log(f"Generating {codebase_type.value} codebase (threaded)", "INFO")
        
        file_names, baseline_model, file_states_attr = self.get_codebase_settings(codebase_type)
        prompt_settings = self.get_prompt_settings()
        codebase_config = self.load_codebase_config()

        def generate_code_for_file(file_name: str, file_alias: str):
            log(f"Generating code for {file_name}", "INFO")
            try:
                file_config = self.get_file_config(codebase_config, file_alias, file_name)
                project_context = self.format_project_context(prompt_settings, codebase_config)
                code_format = self.format_code_template(prompt_settings, file_config)
                module_prompt = self.format_module_prompt(prompt_settings, project_context, file_config['filename'], code_format)
                messages = (prompt_settings['role'], module_prompt)
                return file_alias, self.generate_code_with_retry(Code, messages), messages
            except Exception as e:
                log(f"Error generating code for {file_name}: {str(e)}", "ERROR")
                raise
        
        baseline_code_dict, message_dict = self.generate_baseline_code_dict(file_names, generate_code_for_file)

        try:
            baseline_codes = baseline_model(**baseline_code_dict)
            file_states = FileState.initialize_file_states_from_baseline_code(baseline_codes, self.state.messages)

            # Update file states with generated messages
            for file_alias, file_state in file_states.items():
                basename = os.path.basename(file_state.filename)
                if basename in message_dict:
                    file_state.messages.insert(0, message_dict[basename])
                    
            setattr(self.state, file_states_attr, file_states)
        except ValidationError as e:
            log(f"Error creating BaselineCode: {e}", "ERROR")
            raise

        return self.state
    
    @staticmethod
    def reset_status_for_source_code(file_state: FileState) -> FileState:
        """
        Reset the test status of all files in the shared state.

        Args:
            status (FileStatus): The new test status to set.

        Returns:
            FileStatus: The new test status.
        """
        if file_state.status == FileStatus.IMPORT_TEST_FAILED:
            file_state.status = FileStatus.IMPORT_TEST_PENDING
            file_state.import_passed = False
            file_state.code_passed = False
            file_state.pytest_passed = False
        elif file_state.status == FileStatus.CODE_TEST_FAILED:
            file_state.status = FileStatus.CODE_TEST_PENDING
            file_state.code_passed = False
            file_state.pytest_passed = False
        elif file_state.status == FileStatus.PYTEST_FAILED:
            file_state.status = FileStatus.CODE_TEST_PENDING
            file_state.code_passed = False
            file_state.pytest_passed = False
        file_state.error = ''
        return file_state
    
    @staticmethod
    def reset_status_for_test_code(file_state: FileState) -> FileState:
        """
        Reset the test status of all files in the shared state.

        Args:
            status (FileStatus): The new test status to set.

        Returns:
            FileStatus: The new test status.
        """
        if file_state.status == FileStatus.IMPORT_TEST_FAILED:
            file_state.status = FileStatus.IMPORT_TEST_PENDING
            file_state.import_passed = False
            file_state.code_passed = False
            file_state.pytest_passed = False
        elif file_state.status == FileStatus.PYTEST_FAILED:
            file_state.status = FileStatus.PYTEST_PENDING
            file_state.pytest_passed = False
        file_state.error = ''
        return file_state

    def fix_error_from_source_state_threaded(self, specific_files: Optional[List[str]] = []):
        """
        Generates source code or test code based on the provided schema type.

        Returns:
            SharedState: The updated shared state with the generated source code or test code.
        """
        log(f"Fixing code for Source Code (threaded)", "INFO")

        def fix_code_for_file(name, file_state):
            if name not in specific_files and name in self.test_runner_settings.primary_test_exclusion_files:
                return name, file_state
            # Skip files that have already passed all tests
            if file_state.status == FileStatus.ALL_TESTS_PASSED or not file_state.error:
                return name, file_state
            # Update the file state based on the status
            if file_state.status in [FileStatus.CODE_TEST_FAILED, FileStatus.PYTEST_FAILED]:
                # Generate code for source files
                code = self.generate_code_with_retry(Code, file_state.messages)
                file_state = file_state.update_state_from_code(code, self.max_num_messages)
            elif file_state.status == FileStatus.IMPORT_TEST_FAILED:
                # Generate imports for the file
                code = self.generate_code_with_retry(schema=None, messages=file_state.messages)
                try: 
                    # Extract the imports string from the generated code
                    code = FileHandler.parse_json(code)['solution']
                    file_state.solution['imports'] = code
                    file_state.content = Code.parse_obj(file_state.solution).to_string()
                except (KeyError, ValueError) as e:
                    log(f"Error updating imports for {name}: {e}", "ERROR")
                    log(f"Raw Output: {code}", "INFO")
                    raise e
            # Update file state
            file_state = self.reset_status_for_source_code(file_state)
            return name, file_state
        
        with self.thread_pool as executor:
            if specific_files:
                file_states = [(name, self.state.source_file_states[name]) for name in specific_files]
            else:
                file_states = self.state.source_file_states.items()
            future_to_file = {
                executor.submit(
                    fix_code_for_file,
                    name,
                    file_state): name for name, file_state in file_states
            }
            for future in concurrent.futures.as_completed(future_to_file):
                name, file_state = future.result()
                # Update the appropriate file dictionary with the new file state
                self.state.source_file_states[name] = file_state

        return self.state
    
    def fix_error_from_test_state_threaded(self):
        """
        Generates test code or test code based on the provided schema type.

        Returns:
            SharedState: The updated shared state with the generated source code or test code.
        """
        log(f"Fixing code for Test Code (threaded)", "INFO")

        def fix_code_for_file(name, file_state):
            if name in self.test_runner_settings.primary_test_exclusion_files:
                return name, file_state
            # Skip files that have already passed all tests
            if file_state.status == FileStatus.ALL_TESTS_PASSED:
                return name, file_state
            # Update the file state based on the status
            if file_state.status in [FileStatus.CODE_TEST_FAILED, FileStatus.PYTEST_FAILED]:
                # Generate code for test files
                code = self.generate_code_with_retry(Code, file_state.messages)
                file_state = file_state.update_state_from_code(code, self.max_num_messages)
            elif file_state.status == FileStatus.IMPORT_TEST_FAILED:
                # Generate imports for the file
                code = self.generate_code_with_retry(schema=None, messages=file_state.messages)
                try:
                    # Extract the imports string from the generated code
                    code = FileHandler.parse_json(code)['solution']
                    file_state.solution['imports'] = code
                    file_state.content = Code.parse_obj(file_state.solution).to_string()
                except (KeyError, ValueError) as e:
                    log(f"Error updating imports for {name}: {e}", "ERROR")
                    log(f"Raw Output: {code}", "INFO")
                    raise e
            # Update file state
            file_state = self.reset_status_for_test_code(file_state)

            return name, file_state
        
        with self.thread_pool as executor:
            future_to_file = {
                executor.submit(
                    fix_code_for_file,
                    name,
                    file_state): name for name, file_state in self.state.test_file_states.items()
            }
            for future in concurrent.futures.as_completed(future_to_file):
                name, file_state = future.result()
                # Update the appropriate file dictionary with the new file state
                self.state.test_file_states[name] = file_state

        return self.state
 
    def get_test_prompt_params(self, name, source_file_state, target_filename, module_config):
        test_config = self.load_project_config().get('tests', {}).get(name, {})
        return {
            "module_name": source_file_state.filename,
            "source_code": source_file_state.content,
            "pytest_config": self.get_pytest_config(),
            "conftest": self.get_conftest_content(),
            "test_scenarios": test_config.get('test_scenarios', ''),
            "function_names": test_config.get('function_names', ''),
            "filename": target_filename
        }
    
    def load_project_config(self):
        config_path = self.prompt_settings.config_file_path
        return FileHandler.load_yaml_config(config_path)
    
    def get_test_prompt_settings(self):
        prompt_filename = self.prompt_settings.template_file_path
        prompt_name = self.prompt_settings.prompt_name_mapping['generate_test_code']['test']
        role, prompt = PromptHandler.get_prompt(prompt_filename, prompt_name)
        return {'role': role, 'prompt': prompt}

    def get_pytest_config(self):
        config = self.load_project_config()
        return config.get('pytest_config', {}).get('content', '')

    def get_conftest_content(self):
        config = self.load_project_config()
        return config.get('conftest', {}).get('content', '')

    def format_test_messages(self, prompt_settings, prompt_params):
        """
        Format the test generation prompt with the given parameters.

        Args:
            prompt_settings (dict): Contains the role and prompt template.
            prompt_params (dict): Parameters to fill in the prompt template.

        Returns:
            list: A list containing a tuple of (role, formatted_prompt).
        """
        formatted_prompt = PromptHandler.format_prompt(prompt_settings['prompt'], prompt_params)
        return [(prompt_settings['role'], formatted_prompt)]

    def generate_test_from_source_state_threaded(self) -> SharedState:
        """
        Generate test files based on the source code files and predefined messages.

        Returns:
            SharedState: The updated SharedState object with the generated test file states.
        """
        log("Generating tests from source code (threaded)", "INFO")
        test_names = self.state.baseline_test_file_names.keys()
        prompt_settings = self.get_test_prompt_settings()
        config = self.load_project_config()
        
        def generate_test_file(name):
            try:
                source_file_state = self.state.source_file_states[name]
                target_filename = f'tests/test_{name}.py'
                
                test_config = config['tests'].get(name, {})
                prompt_params = self.get_test_prompt_params(name, source_file_state, target_filename, test_config)
                
                # Check if all required parameters are present
                has_required_params, missing_params = PromptHandler.check_prompt_params(prompt_settings['prompt'], prompt_params)
                if not has_required_params:
                    raise ValueError(f"Missing required parameters for {name}: {', '.join(missing_params)}")
                
                messages = self.format_test_messages(prompt_settings, prompt_params)
                code = self.generate_code_with_retry(Code, messages)
                
                file_state = FileState(
                    filename=target_filename,
                    status=FileStatus.IMPORT_TEST_PENDING,
                    file_type=FileType.TEST,
                    messages=messages
                )
                file_state = file_state.update_state_from_code(code, self.max_num_messages)
                return name, file_state
            except Exception as e:
                log(f"Error generating test file for {name}: {str(e)}", "ERROR")
                raise
        
        with self.thread_pool as executor:
            future_to_file = {executor.submit(generate_test_file, name): name for name in test_names}
            for future in concurrent.futures.as_completed(future_to_file):
                try:
                    name, file_state = future.result()
                    self.state.test_file_states[name] = file_state
                except Exception as e:
                    log(f"Error processing test file result: {str(e)}", "ERROR")

        return self.state
