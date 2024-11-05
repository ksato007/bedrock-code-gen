import re
import boto3
from botocore.exceptions import ClientError
from botocore.config import Config
from .logger import log
from langchain_aws import ChatBedrock

class ModelAvailabilityChecker:
    """
    The ModelAvailabilityChecker class contains static methods to check the availability and access of foundation models in Amazon Bedrock
    """
    @staticmethod
    def get_model_list(model_provider, region):
        """
        Retrieve the list of models from the provided data.
        Args:
            model_provider (str): The provider of the foundation model.
            region (str): The AWS region to check for the model availability.

        Returns:
            list: A list of model summaries.
        """
        try:
            # Create a Bedrock client for the specified region
            bedrock = boto3.client('bedrock', region_name=region)
            # Get the model list in specified region, provider
            response = bedrock.list_foundation_models(byProvider=model_provider)
            model_summaries = response.get('modelSummaries', )
            model_list = []
            for model_summary in model_summaries:
                model_info = {
                    'modelArn': model_summary['modelArn'],
                    'modelId': model_summary['modelId'],
                    'modelName': model_summary['modelName'],
                    'providerName': model_summary['providerName'],
                    'inputModalities': model_summary['inputModalities'],
                    'outputModalities': model_summary['outputModalities'],
                    'responseStreamingSupported': model_summary['responseStreamingSupported'],
                    'customizationsSupported': model_summary['customizationsSupported'],
                    'inferenceTypesSupported': model_summary['inferenceTypesSupported'],
                    'modelLifecycleStatus': model_summary['modelLifecycle'],
                }
                model_list.append(model_info)
            return model_list
        except Exception as e:
            log(f"Error retrieving model list: {e}", level='ERROR')
            return []
        
    @staticmethod
    def has_model_access(model_id, region):
        """
        Check if the specified model has access in the given AWS region.

        Args:
            model_id (str): The ID of the model to check.
            region (str): The AWS region to check the model access in.

        Returns:
            bool: True if the model has access, False otherwise.
        """
        try:
            # Create a Bedrock client for the specified region
            bedrock = boto3.client('bedrock-runtime', region_name=region)
            # Check the model access
            import json
            messages = [{"role": "user", "content": [{"text": "Hello World!"}]}]
            response = bedrock.converse(messages=messages, modelId=model_id)
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                log(f"Model {model_id} has access.", level='INFO')
                return True
        except ClientError as e:
            log(f"Error checking model access: {e}", level='ERROR')
        return False
    @staticmethod
    def is_model_api_available(region, model_provider, model_name, model_version) -> bool:
        """
        Checks if the specified model API is available in the given AWS region.

        Args:
            region (str): The AWS region to check.
            model_provider (str): model provider (eg. "anthropic")
            model_name (str): model_name (e.g., "claude-3-haiku").
            model_version (str): model_version (e.g., "1.x).
        Returns:
            bool: True if the model API is available, False otherwise.
        """
        try:
            # Create a Bedrock client for the specified region
            bedrock = boto3.client('bedrock', region_name=region)

            # List the available models
            response = bedrock.list_foundation_models(byProvider=model_provider)
            log(f"list model reponse: {response}", level='INFO')
            a_model_ver = re.findall(r'\d+', model_version)[0]
            a_minor_ver = re.findall(r'(?<=\d.)\d+', model_version)[0]
            log(f"user specified model version: major={a_model_ver}, minor={a_minor_ver}", level='DEBUG')
            # Check if the specified model is in the list of available models          
            for summary in response["modelSummaries"]:
                b_model_ver  = re.findall(r'(?<=-v)\d+|$', summary["modelId"])[0]
                b_minor_ver  = re.findall(r'(?<=-v\d:)\d+|$', summary["modelId"])[0]
                if a_minor_ver:
                    if (summary["modelId"].startswith(f"{model_provider}.{model_name}") and 
                        a_model_ver.__eq__(b_model_ver) and
                        a_minor_ver.__eq__(b_minor_ver)):
                        return True
                else:
                    if (summary["modelId"].startswith(f"{model_provider}.{model_name}") and 
                        a_model_ver.__eq__(b_model_ver)):
                        return True
            return False
        except ClientError as e:
            # Handle any errors that occur during the API call
            log(f"Error checking model availability: {e}", level='ERROR')
            return False
        
class ModelManager:
    def __init__(self, model_id, model_kwargs, tools=None):
        """
        Initialize the ModelManager instance.

        Args:
            model_id (str): The ID of the model to be used.
            model_kwargs (dict): Additional keyword arguments for the model.
            tools (list, optional): A list of tools to be bound to the model.

        Attributes:
            model_id (str): The ID of the model to be used.
            model_kwargs (dict): Additional keyword arguments for the model.
        """
        self.model_id = model_id
        self.model_kwargs = model_kwargs
        self.tools = tools or []
    def create_model(self):
        """
        Create and initialize the ChatBedrock model.

        Returns:
            ChatBedrock: The initialized ChatBedrock model instance.
        """
        # Create a boto3 client for Bedrock with specified timeout duration
        config = Config(
            read_timeout=300,
            connect_timeout=300
        )
        client = boto3.client('bedrock-runtime', config=config)
        # Create a new instance of the ChatBedrock model
        model = ChatBedrock(
            model_id=self.model_id, 
            model_kwargs=self.model_kwargs,
            client=client
        )
        # Bind the specified tools to the model
        model.bind_tools(self.tools)
        return model
    def check_model_access(self, region):
        """
        Check if the user has access to the specified model in the given region.

        Args:
            region (str): The AWS region to check for model access.

        Returns:
            bool: True if the user has access to the model, False otherwise.
        """
        # Check if the user has access to the model in the specified region
        has_access = ModelAvailabilityChecker.has_model_access(self.model_id, region)
        if not(has_access):
            console_url = f"https://{region}.console.aws.amazon.com/bedrock/home?region={region}#/modelaccess"
            log(f"You don't seem to have access to the model with ID: {self.model_id}", "WARNING")
            log(f"Click here to open the Bedrock console: \033[94m\033[4m{console_url}\033[0m", "INFO")
            return False
        return True