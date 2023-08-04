import numpy as np
from torch.utils.data import DataLoader
from dataclasses import dataclass
from .causal_dataset import CausalDataset
import lightning as L
import os

@dataclass
class BenchmarksConfig:
    data_dir: str = '/mnt/d/benchmarks/tiny_stories/'
    cntx: int = 32
    batch_size: int = 32
    shuffle: bool = True 
    num_workers: int = 4
    pin_memory: bool = True
    lr: float = 1e-3
    warmup_steps: int = 20
    total_steps: int = 30
    cycles: int = 3

class BenchmarksDataModule(L.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        self.train = np.memmap(os.path.join(self.config.data_dir, 'train.tokens'),dtype='uint16', mode='r')
        self.val = np.memmap(os.path.join(self.config.data_dir, 'test.tokens'),dtype='uint16', mode='r')
    
    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        return DataLoader(
            CausalDataset(self.train, self.config.cntx),
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )
    
    def val_dataloader(self):
        return DataLoader(
            CausalDataset(self.val, self.config.cntx),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory
        )