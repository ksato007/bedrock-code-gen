from typing import Optional, List, Dict, Union
from pydantic import BaseModel, ConfigDict
from pydantic_settings import BaseSettings
import yaml

def _import_file_state():
    # lazy imports
    from .graph_state import FileState
    return FileState

class ModelSettings(BaseModel):
    region: str
    model_provider: str
    model_provider_path: str
    model_provider_prefix: str
    model_id: str
    inference_modifier: dict
    tools: list
    max_retries: int
    retry_delay: int

    model_config = ConfigDict(protected_namespaces=())

class PromptSettings(BaseModel):
    template_file_path: str
    config_file_path: str
    prompt_name_mapping: Dict[str, Union[str, Dict[str, str]]]
    max_num_messages: int

class DagSettings(BaseModel):
    checkpoint_path: str
    intermediate_state_file_prefix: str

class TestRunnerSettings(BaseModel):
    generate_test_key: str
    fix_import_key: str
    fix_code_key: str
    fix_pytest_key: str
    primary_test_exclusion_files: List[str]
    directory_tree_dir_exclusion_list: List[str]
    directory_tree_file_inclusion_list: List[str]

class ConfigManager(BaseSettings):
    version: str
    env_settings: dict
    bedrock_model_settings: ModelSettings
    prompt_settings: PromptSettings
    test_runner_settings: TestRunnerSettings
    dag_settings: DagSettings

    class Config:
        # Exclude the 'model_' namespace from protected namespaces
        model_config = ConfigDict(
            env_file = '.env',
            env_file_encoding = 'utf-8',
            env_prefix = 'APP_',
        )

    @classmethod
    def from_env(cls, env_file: Optional[str] = None):
        if env_file:
            return cls(_env_file=env_file)
        else:
            return cls()

    @classmethod
    def from_yaml_file(cls, yaml_file: str):
        with open(yaml_file, 'r') as file:
            config_data = yaml.safe_load(file)
        env_settings = config_data.get('env_settings', {})
        
        # Process venv_name if it exists
        if 'venv_name' in env_settings:
            if not env_settings['venv_name'].endswith('_venv'):
                env_settings['venv_name'] += '_venv'

        bedrock_model_settings = ModelSettings(**config_data.get('bedrock_model_settings', {}))
        prompt_settings = PromptSettings(**config_data.get('prompt_settings', {}))
        test_runner_settings = TestRunnerSettings(**config_data.get('test_runner_settings', {}))
        dag_settings = DagSettings(**config_data.get('dag_settings', {}))

        return cls(
            version=config_data.get('version', '1.0.0'),
            env_settings=env_settings,
            bedrock_model_settings=bedrock_model_settings,
            prompt_settings=prompt_settings,
            test_runner_settings=test_runner_settings,
            dag_settings=dag_settings
        )