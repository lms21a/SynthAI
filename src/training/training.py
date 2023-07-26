import torch
import numpy as np
from datasets import load_dataset
from dataclasses import dataclass
import tiktoken

@dataclass
class TrainArgs:
    test_size: float = 0.005
    tokenizer: str = 'gpt2'
    cntx: int = 64
    btch: int = 64




configs = TrainArgs()
# Preprocessing of data

ds = load_dataset("Skylion007/openwebtext",cache_dir='/mnt/d/benchmarks')
split_dataset = ds['train'].train_test_split(test_size=configs.test_size)
train_dataset = split_dataset['train']
test_dataset = split_dataset['test']

# datasharding
def shard_dataset(dataset, num_shards):
    for i in range(num_shards):
        yield dataset.shard(num_shards, i)

shard_gen = shard_dataset(train_dataset, 500)

# tokenization
enc = tiktoken.encoding_for_model(configs.tokenizer)

def tokenize_shard(shard):
    tokens = [enc.encode_batch(data['text']) for data in shard]
    return tokens

tokenized_shard = tokenize_shard(next(shard_gen))


# Get a batch of data
def get_batch(data,cntx,btch):
    len_data = len(data)-cntx  

    # Generate all starting indices at once
    start_indices = torch.randint(len_data, (btch,)).tolist()

    x = torch.tensor(np.array([data[i:i+cntx] for i in start_indices])).pin_memory().to(device, non_blocking=True)
    y = torch.tensor(np.array([data[i+1:i+cntx+1] for i in start_indices])).pin_memory().to(device, non_blocking=True)
    
    return x, y

print(len(tokenized_shard))


