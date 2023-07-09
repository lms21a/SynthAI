from pydantic import BaseModel, Field
from typing import Any, Optional
import os

class UniversalData(BaseModel):
    data: Any
    metadata: dict = Field(default_factory=dict)

    @staticmethod
    def create(data: Any, binary: Optional[bool] = False) -> 'UniversalData':
        if isinstance(data, str):
            # Check if it's a file path
            if os.path.isfile(data):
                return FilePathData(data=data, binary=binary)
            else:
                return StringData(data=data)
        elif isinstance(data, list):
            return ListData(data=data)
        elif isinstance(data, bytes):
            return BytesData(data=data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

    def compute_metadata(self, data_type: str, is_from_file: bool = False):
        size_mb = len(self.data) / (1024 * 1024)  # convert to MB
        data_class = "toy_dataset" if size_mb < 5 else "standard_dataset"
        self.metadata = {
            "data_type": data_type,
            "length": len(self.data),
            "is_from_file": is_from_file,
            "file_size_MB": size_mb,
            "data_class": data_class
        }

class StringData(UniversalData):
    def __init__(self, **data):
        super().__init__(**data)
        self.data = data.get('data')
        self.compute_metadata(data_type='string')

class ListData(UniversalData):
    def __init__(self, **data):
        super().__init__(**data)
        self.data = data.get('data')
        self.compute_metadata(data_type='list')

class BytesData(UniversalData):
    def __init__(self, **data):
        super().__init__(**data)
        self.data = list(data.get('data'))  # Convert bytes to list of integers
        self.compute_metadata(data_type='bytes')

class FilePathData(UniversalData):
    def __init__(self, **data):
        super().__init__(**data)
        file_path = data.get('data')
        binary = data.get('binary', False)

        # Read file content
        if binary:
            with open(file_path, 'rb') as file:
                file_content = file.read()
            self.data = list(file_content)
            self.compute_metadata(data_type='bytes', is_from_file=True)
        else:
            with open(file_path, 'r') as file:
                file_content = file.read()
            self.data = file_content
            self.compute_metadata(data_type='string', is_from_file=True)

# TODO: Make a "Recipe" class that would create a full stack to train a model based on chosen datasets
# TODO: Add more file types like a generator, csv, pandas dataframe, etc.
