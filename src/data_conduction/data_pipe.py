import torch
from torch.utils.data import Dataset, DataLoader, random_split
import tiktoken
from .universal_data import UniversalData
from ..tools import to_namespace
import json

class CausalDataset(Dataset):
    def __init__(self, data, cntx_len):
        self.data = torch.tensor(data)
        self.cntx_len = cntx_len

    def __len__(self):
        return len(self.data) - self.cntx_len

    def __getitem__(self, index):
        x = self.data[index:index+self.cntx_len]
        y = self.data[index+1:index+self.cntx_len+1]
        return x,y

def pipe(batch_size,context_length, shuffle):
    with open("data/toy_datasets/shakespeare.txt",'r') as file:
        raw_data = file.read()

    data = to_namespace(json.loads(UniversalData.create(raw_data).model_dump_json()))
# TODO: Here we can save as "save_data"
    if data.metadata.data_type == 'list': 
        data.data = "".join(data.data)
        assert(isinstance(data.data,str)), 'ERROR in UniversalData: Data is not a string'

    if data.metadata.data_type != 'bytes':
        enc = tiktoken.encoding_for_model('gpt2')
        data.data = enc.encode(data.data)
# TODO: We can save here as "save_tokens"

# Train Test Split
    dataset = CausalDataset(data.data,cntx_len=context_length)
    # TODO: Remove Later
    train_size = int(.7*len(dataset))
    val_size = int(.2*len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset,[train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=shuffle)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=shuffle)
    test_loader = DataLoader(test_dataset,batch_size=batch_size,shuffle=shuffle)

# x, y = next(iter(val_loader))
# print(x)
# print(y)
    return train_loader, val_loader, test_loader

