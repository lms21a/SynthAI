from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from .components import RMSNorm, LlamaKinda_Block, LlamaKindaArgs, precompute_freqs_cis
from torch.nn import functional as F

class LlamaKinda(nn.Module):
    def __init__(self, args: LlamaKindaArgs):
        super().__init__()
        
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

# dummy = torch.randint(0, 50257, (32, 32))
# model = LlamaKinda(LlamaKindaArgs())
# print(model(dummy).shape)