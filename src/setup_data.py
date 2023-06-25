import torch
import tiktoken
from torch.utils.data import DataLoader, Dataset
from torch.utils.data import Dataset
from unit_tests import check_for_causal

UNIT_TEST = False

class CausalDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = data
        self.context_length = context_length
        self.inputs = self.data[:-1].unfold(0, self.context_length, 1)
        self.targets = self.data[1:].unfold(0, self.context_length, 1)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return self.inputs.size(0)
    
    def _split(self,train_size=.7,valid_size=.2):
        train_size = int(train_size * len(self))
        valid_size = int(valid_size * len(self))
        test_size = len(self) - train_size - valid_size
        if test_size == 0:
            return torch.utils.data.random_split(self, [train_size, valid_size]) # type: ignore
        return torch.utils.data.random_split(self, [train_size, valid_size, test_size]) # type: ignore

    def return_dataloaders(self,batch_size,shuffle):
        train,val,test = self._split()
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=shuffle)
        return train_loader,val_loader,test_loader

def preprocess(context_length,batch_size,shuffle):
    enc= tiktoken.encoding_for_model('gpt2')
    with open('synthai/data/shakespeare.txt', 'r') as file:
        text = file.read()

    data = torch.tensor(enc.encode(text), dtype=torch.long)


    dataset = CausalDataset(data, context_length)
    train_loader,val_loader,test_loader= dataset.return_dataloaders(batch_size=batch_size,shuffle=shuffle)
    if UNIT_TEST:
        check_for_causal(train_loader)
        check_for_causal(val_loader)
        check_for_causal(test_loader)

    return train_loader,val_loader,test_loader

# check_for_causal(train_loader) # check_for_causal(val_loader)
# check_for_causal(test_loader)