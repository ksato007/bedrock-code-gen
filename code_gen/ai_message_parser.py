from typing import (
    Dict, 
    Tuple,
    Type,
    Union
)
from langchain_core.pydantic_v1 import (
    BaseModel,
)
from langchain_core.messages import (
    AIMessage,
)
from .logger import log
import re 
import xmltodict
# import xml.etree.ElementTree as ET
# from xml.parsers.expat import ExpatError
from defusedxml import ElementTree as secure_ET
from defusedxml.common import DefusedXmlException
import io
from xml.parsers.expat import ExpatError

class AIMessageParser:
    @staticmethod
    def is_xml_complete(xml_string):
        try:
            # the secure parsing is already built into the defusedxml fromstring() function            
            secure_ET.fromstring(xml_string)
            return True
        except secure_ET.ParseError as e:
            if "no element found" in str(e):
                return False
            elif "mismatched tag" in str(e):
                return AIMessageParser._lenient_xml_check(xml_string)
            else:
                return True
        except DefusedXmlException:
            return False

    @staticmethod
    def _lenient_xml_check(xml_string):
        # Count opening and closing tags
        opening_tags = xml_string.count('<') - xml_string.count('</')
        closing_tags = xml_string.count('</') + xml_string.count('/>')
        
        # Check if we have equal numbers of opening and closing tags
        if opening_tags == closing_tags:
            return True
        
        # Check if the last non-whitespace character is '>'
        stripped = xml_string.strip()
        if stripped and stripped[-1] == '>':
            return True
        
        return False

    @staticmethod
    def extract_elements(content: str, schema_class: Type[BaseModel]):
        try:
            wrapped_content = f'<root>{content}</root>'
            # Use defusedxml's fromstring without the parser argument
            root = secure_ET.fromstring(wrapped_content)
        except (secure_ET.ParseError, DefusedXmlException) as e:
            raise ValueError(f"XML parsing error: {str(e)}")

        parameters = root.find('.//parameters')
        if parameters is None:
            return {}
        else:
            parameter_values = {}
            schema_keys = schema_class.schema()['properties'].keys()
            for k in schema_keys:
                param = parameters.find(k)
                if param is not None:
                    parameter_values[k] = param.text
            return parameter_values

    @staticmethod
    def remove_duplicate_elements(wrapped_content):
        try:
            # Use defusedxml's fromstring without the parser argument
            root = secure_ET.fromstring(wrapped_content)
        except (secure_ET.ParseError, DefusedXmlException) as e:
            raise ValueError(f"XML parsing error: {str(e)}")
        
        # Function to recursively remove duplicates
        def remove_duplicates(element):
            # Keep track of elements we've seen
            seen_elements = set()
            children = list(element)
            for child in children:
                if child.tag in seen_elements:
                    element.remove(child)
                else:
                    seen_elements.add(child.tag)
                    remove_duplicates(child)
        
        # Remove duplicates starting from the root
        remove_duplicates(root)
        
        # Convert back to string
        return secure_ET.tostring(root, encoding='unicode')

    @staticmethod
    def wrap_code_in_cdata(content):
        """
        Wrap code snippets in the given content with CDATA tags.

        This function takes a string `content` and replaces all occurrences of
        `<code>...</code>` with `<code><![CDATA[...]]></code>`. This is useful
        when dealing with XML content that contains code snippets, as it ensures
        that the code snippets are treated as character data and not parsed as XML.

        Args:
            content (str): The input string containing code snippets.

        Returns:
            str: The input string with code snippets wrapped in CDATA tags.
        """
        def replace_code(match):
            """
            Replace a matched code snippet with a CDATA-wrapped version.

            Args:
                match (re.Match): The regular expression match object.

            Returns:
                str: The CDATA-wrapped code snippet.
            """
            # Extract the code snippet from the match
            code = match.group(1)
            # Wrap the code with CDATA tags
            return f'<code><![CDATA[{code}]]></code>'
        
        return re.sub(r'<code>(.*?)</code>', replace_code, content, flags=re.DOTALL)

    @staticmethod
    def xml_to_dict(ai_message: AIMessage) -> Dict:
        # Wrap the content in a root element
        wrapped_content = f'<root>{ai_message.content}</root>'

        try:
            # Use defusedxml to parse the XML
            tree = secure_ET.parse(io.StringIO(wrapped_content))
            
            # Convert the parsed XML to a string
            safe_xml = secure_ET.tostring(tree.getroot(), encoding='unicode')
            
            # Now use xmltodict on the safe XML string
            parsed_data = xmltodict.parse(safe_xml)
        except (DefusedXmlException, ExpatError, ValueError, TypeError) as e:
            log(f"Invalid XML format in AIMessage: {ai_message.content}", "ERROR")
            raise ValueError(f"Error parsing XML: {str(e)}")

        if 'root' not in parsed_data:
            raise ValueError("Parsed XML does not contain expected 'root' element")
        
        return parsed_data['root']

    @staticmethod
    def parse_ai_message(ai_message: AIMessage, schema_class: Type[BaseModel]) -> Type[BaseModel]:
        """
        Parse an AI message and return a Pydantic BaseModel instance representing the structured output.

        Args:
            ai_message (AIMessage): The AI message to be parsed.
            schema_class (Type[BaseModel]): The Pydantic BaseModel class that defines the
                expected structure of the AI message output.

        Returns:
            Type[BaseModel]: A Pydantic BaseModel instance created from the parsed data.
        """
        # escape special characters in Python
        ai_message.content = AIMessageParser.wrap_code_in_cdata(ai_message.content)
        # check if it contains a valid xml string
        if not AIMessageParser.is_xml_complete(f'<root>{ai_message.content}</root>'):
            log(f"Invalid XML format in AIMessage: {ai_message.content}", "ERROR")
            raise ValueError("Invalid XML format")
        # check tooling calling function used
        parsed_data =  AIMessageParser.extract_elements(
            content=ai_message.content, 
            schema_class=schema_class
        )
        if parsed_data:
            return schema_class.parse_obj(parsed_data)
        # convert xml and run schema parsing 
        parsed_data = AIMessageParser.xml_to_dict(ai_message)
        return schema_class.parse_obj(parsed_data)
    
    @staticmethod
    def has_html_tags(ai_message: AIMessage) -> bool:
        """
        Check if the AI message content contains any HTML tags.

        Args:
            ai_message (AIMessage): The AI message to be checked.

        Returns:
            bool: True if the AI message content contains HTML tags, False otherwise.
        """
        return len(re.findall(r"<([^>]*)>", ai_message.content)) > 0
    @staticmethod
    def get_structured_ouput(schema: Type[BaseModel]):
        """
        This is a factory method that returns a function to parse an AI message and
        return a structured output based on the provided Pydantic BaseModel schema.

        Args:
            schema (Type[BaseModel]): The Pydantic BaseModel class that defines the
                expected structure of the AI message output.

        Returns:
            Callable[[AIMessage], Union[Type[BaseModel], str]]: A function that takes an AIMessage
                instance and returns either a Pydantic BaseModel instance representing the
                structured output or a string if the text does not require parsing.
        """
        def parse_output(ai_message: AIMessage) -> Union[Type[BaseModel], str]:
            """
            Parse an AI message and return a structured output based on the provided
            Pydantic BaseModel schema or return the text as-is if it does not require parsing.

            This function checks the content of the `AIMessage` and determines whether it
            contains structured data that needs to be parsed using the provided schema or
            if it is plain text that should be returned as a string.

            Args:
                ai_message (AIMessage): The AI message to be parsed.

            Returns:
            Union[Type[BaseModel], str]: A Pydantic BaseModel instance created from the
                parsed data or a string if the text does not require parsing.
            """
            # log(f"AI Message: {ai_message.content}", "DEBUG")
            # Check if normal text or contain structured data
            if AIMessageParser.has_html_tags(ai_message):
                return AIMessageParser.parse_ai_message(ai_message=ai_message, schema_class=schema)
            else:
                # Return the content as a string if it does not require parsing
                ai_message.content
        return parse_output