import json
import types
import tiktoken
import matplotlib.pyplot as plt

def to_namespace(json_object): 
    if isinstance(json_object,dict):
        namespace_dict = {}
        for key,value in json_object.items():
            namespace_dict[key] = to_namespace(value)
        return types.SimpleNamespace(**namespace_dict)
    elif isinstance(json_object,list):
        return [to_namespace(item) for item in json_object]
    else:
        return json_object

def read_json_config(file_path) -> types.SimpleNamespace:
    """
    Read data from a JSON file and convert it to a SimpleNamespace object.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        types.SimpleNamespace: A SimpleNamespace object containing the data read from the JSON file.
    """
    with open(file_path, "r") as json_file:
        config_data = json.load(json_file)
    return to_namespace(config_data)

def convert_readable(generated_output, enc = tiktoken.encoding_for_model('gpt2')):
    return enc.decode_batch(generated_output.tolist())

def print_pipe(data_pipe):
    for name, data in data_pipe:
        print(name,data[:10])