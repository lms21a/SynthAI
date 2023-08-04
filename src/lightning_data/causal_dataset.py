import torch
import numpy as np
import math

# https://github.com/Dao-AILab/flash-attention/blob/main/training/src/datamodules/datasets/lm_dataset.py
class CausalDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, cntx, drop_last=True):
        self.cntx = cntx
        ntokens = len(tokens)
        if drop_last:
            ntokens = ((ntokens - 1) // cntx) * cntx + 1
        self.ntokens = ntokens
        self.tokens = tokens
        self.total_sequences = math.ceil((self.ntokens - 1) / self.cntx)

    def __len__(self):
        return self.total_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.cntx
        cntx = min(self.cntx, self.ntokens - 1 - start_idx)
        data = torch.as_tensor(self.tokens[start_idx:(start_idx + cntx + 1)].astype(np.int64))
        return data[:-1], data[1:].clone()