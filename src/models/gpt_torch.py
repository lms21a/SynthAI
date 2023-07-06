import torch
import torch.nn as nn
import torch.nn.functional as F
from .components import Wte_Wpe, GPT_Block_torch
from dataclasses import dataclass

@dataclass
class GPT_Config:
    vocab_size: int = 50257
    d_model: int = 128
    cntx_len: int = 128
    n_layers: int = 2
    n_head: int = 8
    dropout_p: float = 0.0

class GPT_torch(nn.Module):
    def __init__(self, config):
        super(GPT_torch, self).__init__()
        self.config = config
        self.wte_wpe = Wte_Wpe(config.vocab_size, config.d_model, config.cntx_len, config.dropout_p)
        self.blocks = nn.ModuleList([GPT_Block_torch(config.d_model, config.n_head, config.dropout_p) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.d_model)
        self.fc_out = nn.Linear(config.d_model, config.vocab_size, bias=False)
 
    def forward(self, x):
        x = self.wte_wpe(x)
        for block in self.blocks:
            x = block(x)
        return self.fc_out(self.ln(x))

    # https://github.com/karpathy/nanoGPT/blob/master/model.py
    @torch.no_grad()
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