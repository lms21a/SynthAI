import json
import types
import tiktoken
import subprocess
import matplotlib.pyplot as plt
def read_json_config(file_path):
    """
    Read data from a JSON file and convert it to a SimpleNamespace object.

    Args:
        file_path (str): The path to the JSON file to be read.

    Returns:
        types.SimpleNamespace: A SimpleNamespace object containing the data read from the JSON file.
    """
    with open(file_path, "r") as json_file:
        config_data = json.load(json_file)
    return types.SimpleNamespace(**config_data)


def convert_readable(generated_output, enc = tiktoken.encoding_for_model('gpt2')):
    return enc.decode_batch(generated_output.tolist())

def visualize(train_losses,val_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()