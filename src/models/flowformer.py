import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Wte_Wpe, FlowFormer_Block_GLU, FlowFormer_Block_Squared
from dataclasses import dataclass
from ..tools import convert_readable

@dataclass
class FlowFormer_Config:
    vocab_size: int = 50257
    d_model: int = 2048
    cntx_len: int = 32 
    n_layers: int = 15 
    n_head: int = 64
    n_groups: int = 64 
    ff_type: str = 'glu'

class FlowFormer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte_wpe = Wte_Wpe(config.vocab_size, config.d_model, config.cntx_len)
        if config.ff_type == 'glu':
            self.blocks = nn.ModuleList([FlowFormer_Block_GLU(config.d_model, config.n_head, config.n_groups) for _ in range(config.n_layers)])
        else:
            self.blocks = nn.ModuleList([FlowFormer_Block_Squared(config.d_model, config.n_head, config.n_groups) for _ in range(config.n_layers)]) 
        self.ln = nn.LayerNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, x, y = None):
        x = self.wte_wpe(x)
        for block in self.blocks:
            x = block(x)
        logits = self.fc_out(self.ln(x))
        if y is not None:
            loss = F.cross_entropy(logits.view(-1,logits.size(-1)), y.view(-1))
            return logits, loss
        return logits
    
    def print_model_size(self):
        total_params = sum(p.numel() for p in self.parameters())
        formatted_size = "{:,}".format(total_params)
        print(f"Model size: {formatted_size} parameters")
    
    def count_model_memory(self):
        total_memory = 0
        for param in self.parameters():
            # Multiply number of elements in tensor by its byte size
            total_memory += param.nelement() * param.element_size()

        # Convert bytes to megabytes
        total_memory_mb = total_memory / (1024 ** 2)

        # Get total GPU memory and used memory
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        total_gpu_memory_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
        current_gpu_memory_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)

        print(f"Model memory usage: {total_memory_mb:.2f} MB")
        print(f"Current GPU memory usage: {current_gpu_memory_mb:.2f} MB / {total_gpu_memory_mb:.2f} MB")

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at cntx_len
            idx_cond = idx if idx.size(1) <= self.config.cntx_len else idx[:, -self.config.cntx_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx 


