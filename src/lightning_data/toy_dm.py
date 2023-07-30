import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass
from .causal_dataset import CausalDataset
import lightning as L

@dataclass
class ToyConfig:
    data: str = 'data/toy_datasets/lexpods.bin'
    batch_size: int = 32
    cntx: int = 32
    num_workers: int = 4
    lr: float = 0.005 

class ToyDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        self.tokens = np.memmap(self.config.data, mode='r')

    def setup(self, stage=None):
        self.train = self.tokens[:int(0.8*len(self.tokens))]
        self.val = self.tokens[int(0.8 * len(self.tokens)):]
    
    def train_dataloader(self):
        return DataLoader(CausalDataset(self.train, self.config.cntx), batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)
    
    def val_dataloader(self):
        return DataLoader(CausalDataset(self.val, self.config.cntx), batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers)