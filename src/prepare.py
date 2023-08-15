import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os 
import gzip
import shutil
import pickle

dataset = 'Skylion007/openwebtext' # Huggingface dataset name
cache_dir = '/mnt/d/hf_cache/' # Where to chache the huggingface dataset
save_dir = '/mnt/d/benchmarks/openweb/' # Where to save the .tokens files
enc = tiktoken.get_encoding('gpt2') # Encoding to use
dataset_idx = 'text'
test_size = 0.0005 # How large the test set should be 
batch_size = 2048 # How many Rows to process at a time
max_rows = 10000 #shard size
compression = False

def create_file(file_dir, file_name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return os.path.join(file_dir, file_name)

def tokenize(data):
    tokens = enc.encode_ordinary_batch(data[dataset_idx])
    for t in tokens:
        t.append(enc.eot_token)
    lengths = [len(t) for t in tokens]  # calculate the length of each tokenized example
    return {'tokens': tokens, 'len': lengths}

def compress_folder(directory):
    files_in_directory = os.listdir(directory)
    
    for filename in tqdm(files_in_directory,desc='Compressing Files'):
        file_path = os.path.join(directory, filename)
        
        with open(file_path, 'rb') as f_in:
            compressed_file_path = file_path + '.gz'
            
            with gzip.open(compressed_file_path, 'wb') as f_out:
                # Copy the contents of the original file to the compressed file
                shutil.copyfileobj(f_in, f_out)
                
        os.remove(file_path)
    print("Compressed All Files")

def decompress_folder(directory):
    files_in_directory = os.listdir(directory)
    for filename in tqdm(files_in_directory,desc='Decompressing Files'):
        if filename.endswith('.gz'):
            file_path = os.path.join(directory, filename)
            
            with gzip.open(file_path, 'rb') as f_in:
                # Construct the name of the decompressed file
                decompressed_file_path = file_path[:-3]
                
                with open(decompressed_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                    
            os.remove(file_path)
    print("Decompressed All Files")

def save_dict_pickle(dictionary, filename):
    with open(filename, "wb") as f:
        pickle.dump(dictionary, f)
    print(f"Dictionary saved to {filename}")

def load_dict_pickle(filename):
    with open(filename, "rb") as f:
        loaded_dict = pickle.load(f)
    return loaded_dict

ds = load_dataset(dataset, cache_dir=cache_dir)

split_dataset = ds['train'].train_test_split(test_size=test_size)

tokenized_dataset = split_dataset.map(
    tokenize,
    remove_columns=split_dataset['train'].column_names,
    desc='Tokenizing',
    num_proc=8,
    batched=True,
    batch_size=batch_size
)

meta = {}
for split, data in tqdm(tokenized_dataset.items(),desc='Saving'):
    total_tokens = np.sum(data['len'])
    meta[f'{split}_len'] = total_tokens
    filename = create_file(save_dir,f'{split}.tokens')
    memarr = np.memmap(filename, mode='w+', dtype=np.uint16, shape=(total_tokens,))
    num_shards = data.num_rows // max_rows if data.num_rows > max_rows else 1

    idx = 0
    for s in tqdm(range(num_shards),desc='Sharding'):
        shard = data.shard(num_shards=num_shards,index=s,contiguous=True).with_format('numpy')
        data_shard = np.concatenate(shard['tokens'])
        memarr[idx:idx + len(data_shard)] = data_shard
        idx += len(data_shard)

    memarr.flush()

save_dict_pickle(meta, create_file(save_dir,'meta.pkl'))
compress_folder(save_dir) if compression else None