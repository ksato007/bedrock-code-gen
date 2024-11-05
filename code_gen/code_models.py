from pydantic import ValidationError
from langchain_core.pydantic_v1 import BaseModel, Field, create_model
from typing import Optional, Dict
import re

class Code(BaseModel):
    """
    It defines a Pydantic model called Code that represents a code file with its description, steps, imports, code content, and filename
    """
    description: str = Field(
        description="Description of the problem and approach")
    steps: str = Field(
        description="Steps to solve the problem")
    imports: Optional[str] = Field(
        description="Code block import statements")
    code: str = Field(
        description="Code block not including import statements")
    filename: str = Field(
        description="relative path to file")

    def to_string(self) -> str:
        if self.imports:
            return f'''"""\n{self.description}\n\n{self.steps}\n"""\n\n{self.imports}\n\n{self.code}'''.strip()
        else:
            return f'''{self.code}'''.strip()

    def to_string_code(self) -> str:
        if self.imports:
            return f'''\n{self.imports}\n\n{self.code}'''.strip()
        else:
            return f'''{self.code}'''.strip()
            
    class Config:
        # exclude fields with Non when model is serialized
        exclude_unset = True

def create_baseline_code_model(file_list: Dict[str, str]):
    """
    Create a BaselineCode model dynamically based on the provided file list.
    example usage:
    model = create_baseline_code_model({'preprocess.py': 'preprocess', 'model.py': 'model'})
    """
    fields = {}
    # Loop through the file list and create fields for the BaselineCode model
    for file_name, alias in file_list.items():
        fields[file_name] = (Code, Field(description=f"The {file_name} file.", alias=alias))

    return create_model(
        "BaselineCode",
        **fields,
    )

def parse_file_content(content):
    description = re.search(r'<description>(.*?)</description>', content, re.DOTALL).group(1).strip()
    steps = re.search(r'<steps>(.*?)</steps>', content, re.DOTALL).group(1).strip()
    imports = re.search(r'<imports>(.*?)</imports>', content, re.DOTALL).group(1).strip()
    code = re.search(r'<code><!\[CDATA\[(.*?)\]\]></code>', content, re.DOTALL).group(1).strip()
    filename = re.search(r'<filename>(.*?)</filename>', content, re.DOTALL).group(1).strip()
    
    return Code(
        description=description,
        steps=steps,
        imports=imports,
        code=code,
        filename=filename
    )