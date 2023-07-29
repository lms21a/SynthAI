import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os 
import gzip
import shutil

dataset = 'bigcode/the-stack-dedup'
data_dir = 'data/python'
dataset_idx = 'content'
cache_dir = '/mnt/d/hf_cache/'
save_dir = '/mnt/d/benchmarks/stack_python/'
test_size = 0.0005
enc = tiktoken.get_encoding('gpt2')
max_rows = 10000 #shard size
compression = False
iters = 4

def create_file(file_dir, file_name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return os.path.join(file_dir, file_name)

def tokenize_data(data):
    ids = enc.encode(data[dataset_idx],allowed_special='all')
    ids.append(enc.eot_token)
    output = {'tokens': ids, 'len': len(ids)}
    return output
    
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

# Move this to a different folder 
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

full_ds = load_dataset(dataset,data_dir=data_dir,cache_dir=cache_dir)

for i in range(iters):
    ds = full_ds['train'].shard(iters,i)
    split_dataset = ds.train_test_split(test_size=test_size)
    preprocessed_dataset = split_dataset.map(
        tokenize_data,
        remove_columns=[dataset_idx],
        desc='Tokenizing',
        num_proc=8
    )
    info = {'train':None,'test':None}
    for split, ds in preprocessed_dataset.items(): 
        max_len = np.sum(ds['len'])
        info[split] = f'Total Tokens: {max_len}'        
        filename = create_file(save_dir,f'{split}_{i}.tokens')
        data = np.memmap(filename,mode='w+',shape=(max_len,))
        num_shards = ds.num_rows // max_rows

        idx = 0
        for s in tqdm(range(num_shards)):
            shard = ds.shard(num_shards=num_shards,index=s,contiguous=True).with_format('numpy')
            data_shard = np.concatenate(shard['tokens'])
            data[idx:idx + len(data_shard)] = data_shard
            idx += len(shard['tokens'])

        data.flush()
        # TODO: Create a sole info file
    info_file = os.path.join(save_dir, f'info{i}.txt')

    with open(info_file, 'w') as f:
        for split in ['train','test']:
            f.write(f'Split: {split}_{i}\n\t{info[split]}_{i}\n')
        f.write('Compression: True\n' if compression else 'Compression: False\n')
    
compress_folder(save_dir) if compression else None
