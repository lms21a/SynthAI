import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
import os 

cache_dir = '/mnt/d/hf_cache'
save_dir = '/mnt/d/benchmarks/openweb'
test_size = 0.1
enc = tiktoken.get_encoding('gpt2')
max_rows = 10000

ds = load_dataset("Skylion007/openwebtext",cache_dir=cache_dir)

split_dataset = ds['train'].train_test_split(test_size=test_size)

def create_file(file_dir, file_name):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    return os.path.join(file_dir, file_name)

def tokenize_data(data):
    ids = enc.encode_ordinary(data['text'])
    ids.append(enc.eot_token)
    output = {'tokens': ids, 'len': len(ids),'max': max(ids)} # Used to minimize the vocab size
    return output

preprocessed_dataset = split_dataset.map(
    tokenize_data,
    remove_columns=['text'],
    desc='Tokenizing',
    num_proc=12
)

info = {'train':None,'test':None}

for split, ds in preprocessed_dataset.items(): 
    max_len = np.sum(ds['len'])
    max_vocab = np.max(ds['max'])
    info[split] = f'Total Tokens: {max_len}\nMax Vocab Size: {max_vocab}'        
    filename = create_file(save_dir,f'{split}.tokens')
    data = np.memmap(filename,mode='w+',shape=(max_len,))
    num_shards = ds.num_rows // max_rows

    idx = 0
    for s in tqdm(range(num_shards)):
        shard = ds.shard(num_shards=num_shards,index=s,contiguous=True).with_format('numpy')
        data_shard = np.concatenate(shard['tokens'])
        data[idx:idx + len(data_shard)] = data_shard
        idx += len(shard['tokens'])

    data.flush()

info_file = os.path.join(save_dir, 'info.txt')
with open(info_file, 'w') as f:
    for split in ['train','test']:
        f.write(f'Split: {split}\n{info[split]}\n')
