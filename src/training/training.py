import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from ..models.gpt_torch import GPT_torch, GPT_Config
from .schedulers import adaptive_momentum_scheduler, update_grad

@dataclass
class TrainArgs:
    lr_init = .1
    momentum = 0
    max_steps = 5000
    batch_size = 4
    eval_interval = 500
    step = 0
    warm_up_steps = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_config = GPT_Config(
    vocab_size=50257,
    d_model = 32,
    cntx_len=32,
    n_layers=2,
    n_head=4
)    

def get_batch(split,btch=TrainArgs.batch_size,cntx=model_config.cntx_len,device=TrainArgs.device):
    data = train if split == 'train' else test 
    len_data = len(data)-cntx  

    # Generate all starting indices at once
    start_indices = torch.randint(len_data, (btch,)).tolist()

    x = torch.tensor(np.array([data[i:i+cntx] for i in start_indices])).pin_memory().to(device, non_blocking=True)
    y = torch.tensor(np.array([data[i+1:i+cntx+1] for i in start_indices])).pin_memory().to(device, non_blocking=True)
    
    return x, y


data = np.memmap('lexpods.bin', mode='r')
train = data[:int(.9*len(data))]
test = data[int(.9*len(data)):]





@torch.inference_mode()
def eval_losses(iters):
    losses = []
    for _ in range(iters):
        x,y = get_batch('valid',TrainArgs.batch_size,model_config.cntx_len)
        loss,_ = model(x,y)
        losses.append(loss.item())
    avg_loss = sum(losses) / len(losses)
    return avg_loss

