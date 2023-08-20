from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from .components import RMSNorm, LlamaKinda_Block, LlamaKindaArgs, precompute_freqs_cis
from torch.nn import functional as F

class LlamaKinda(nn.Module):
    def __init__(self, args: LlamaKindaArgs):
        super().__init__()
        self.cntx = args.max_seq_len        
        self.embed = nn.Embedding(args.vocab_size, args.dim)
        self.drop = nn.Dropout(args.dropout)
        self.layers = nn.ModuleList([LlamaKinda_Block(args) for _ in range(args.n_layers)])
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

        # tie weights
        self.embed.weight = self.lm_head.weight

        # RoPE
        freq_cos, freq_sin = precompute_freqs_cis(args.dim // args.n_qheads, args.max_seq_len)
        self.register_buffer('freq_cos', freq_cos, persistent=False)
        self.register_buffer('freq_sin', freq_sin, persistent=False)

    def forward(self, input: torch.Tensor, target: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = input.shape
        input = self.embed(input)
        input = self.drop(input)
        rope_cos = self.freq_cos[:t]
        rope_sin = self.freq_sin[:t]

        for layer in self.layers:
            input = layer(input, rope_cos, rope_sin)
        input = self.norm(input)

        if target is not None:
            logits = self.lm_head(input)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
            return loss, logits
        
        logits = self.lm_head(input[:, [-1], :])
        return logits
    
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
            idx_cond = idx if idx.size(1) <= self.cntx else idx[:, -self.cntx:]
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
# dummy = torch.randint(0, 50257, (32, 32))
# model = LlamaKinda(LlamaKindaArgs())
# print(model(dummy).shape)