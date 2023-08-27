import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
from einops import rearrange #type: ignore
from einops.layers.torch import Rearrange #type: ignore
from typing import Optional, Tuple
from dataclasses import dataclass

# ------------------- # Traditional Transformer, Pytorch # ------------------- #
class Wte_Wpe(nn.Module):
    def __init__(self, vocab_size, d_model, cntx_len,dropout=0.0):
        super(Wte_Wpe, self).__init__()
        self.d_model = d_model
        self.cntx_len = cntx_len
        self.vocab_size = vocab_size
        self.dropout_p = dropout
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(cntx_len, d_model)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.wte.weight, std=0.02)
        nn.init.normal_(self.wpe.weight, std=0.02)

    def forward(self, x):
        pos = torch.arange(0, x.size(1), dtype=torch.long, device=x.device)
        return F.dropout(self.wte(x)+self.wpe(pos),p=self.dropout_p)
    

class CSA_torch(nn.Module):
    def __init__(self, d_model, n_head,dropout=0.0):
        super(CSA_torch, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        assert d_model % n_head == 0
        self.head_size = d_model // n_head
        self.dropout_p = dropout

        self.qkv = nn.Linear(d_model, 3*d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.init_weights()
    def init_weights(self):
            nn.init.xavier_uniform_(self.qkv.weight)
            nn.init.xavier_uniform_(self.fc_out.weight)
            nn.init.zeros_(self.qkv.bias)
            nn.init.zeros_(self.fc_out.bias)
    def forward(self, x):
        q,k,v = self.qkv(x).split(self.d_model, dim=2)
        q = rearrange(q, 'b t (nh hs) -> b nh t hs', nh=self.n_head, hs=self.head_size)
        k = rearrange(k, 'b t (nh hs) -> b nh t hs', nh=self.n_head, hs=self.head_size)
        v = rearrange(v, 'b t (nh hs) -> b nh t hs', nh=self.n_head, hs=self.head_size)
        y = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.dropout_p, is_causal=True)
        y = rearrange(y, 'b nh t hs -> b t (nh hs)')
        return F.dropout(self.fc_out(y), p=self.dropout_p)

class GPT_Block_torch(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.0):
        super(GPT_Block_torch, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.dropout_p = dropout

        self.ln1 = nn.LayerNorm(d_model)
        self.csa = CSA_torch(d_model,n_head,dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )
        self.init_weights()
    
    def init_weights(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        x = x + self.csa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# ------------------- # LlamaKinda  # ------------------- #
@dataclass
class LlamaKindaArgs:
    dim: int = 64
    n_layers: int = 2
    n_qheads: int = 2
    n_kvheads: Optional[int] = None
    vocab_size: int = 50257
    norm_eps: float = 1e-5
    max_seq_len: int = 64 
    dropout: float = 0.0
    ff_act_fn: nn.Module = nn.GELU()
    multiplier: int = 2

# RoPE Functions
def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

# New Attention. 
# If group_dim == 1, MHA
# If group_dim > 1, GQA
# If group_dim = n_qheads, MQA
def repeat_kv(x: torch.Tensor, group_size: int) -> torch.Tensor:
    b, t, n_kvheads, head_dim = x.shape
    if group_size == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(b, t, n_kvheads, group_size, head_dim)
        .reshape(b, t, n_kvheads * group_size, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: LlamaKindaArgs):
        super().__init__()
        assert args.dim % args.n_qheads == 0, 'Embedding Dimension must be divisible by number of heads'
        if args.n_kvheads is None:
            args.n_kvheads = args.n_qheads
        assert args.n_qheads % args.n_kvheads == 0, 'Number of heads must be divisible by number of key-value heads'

        self.n_qheads = args.n_qheads
        self.n_kvheads = args.n_kvheads
        self.head_dim = args.dim // args.n_qheads
        self.group_dim = args.n_qheads // args.n_kvheads
        self.dropout_p = args.dropout

        self.wq = nn.Linear(args.dim, args.n_qheads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kvheads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kvheads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_qheads * self.head_dim, args.dim, bias=False)

        self.drop = nn.Dropout(args.dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        b, t, _ = x.shape
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        q = q.view(b, t, self.n_qheads, self.head_dim)
        k = k.view(b, t, self.n_kvheads, self.head_dim)
        v = v.view(b, t, self.n_kvheads, self.head_dim)

        q, k = apply_rotary_emb(q, k, rope_cos, rope_sin)

        q = q.transpose(1,2)
        k,v = repeat_kv(k, self.group_dim).transpose(1,2), repeat_kv(v, self.group_dim).transpose(1,2)

        y = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_p, is_causal=True, attn_mask=None)
        y = y.transpose(1,2).contiguous().view(b, t, -1)
        return self.drop(self.wo(y))

# gated linear unit feedforward, but with activation of choice
class ActGLU(nn.Module):
    def __init__(self, args: LlamaKindaArgs):
        super().__init__()
        
        self.w1 = nn.Linear(args.dim, args.dim * args.multiplier, bias=False)
        self.gate = nn.Linear(args.dim, args.dim * args.multiplier, bias=False)
        self.w2 = nn.Linear(args.dim * args.multiplier, args.dim, bias=False)
        self.drop = nn.Dropout(args.dropout)
        self.act = args.ff_act_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w2(self.w1(x) * self.act(self.gate(x))))

class LlamaKinda_Block(nn.Module):
    def __init__(self, args: LlamaKindaArgs):
        super().__init__()
        self.prenorm1 = RMSNorm(args.dim, args.norm_eps)
        self.prenorm2 = RMSNorm(args.dim, args.norm_eps)
        self.attn = Attention(args)
        self.ff = ActGLU(args)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.prenorm1(x), rope_cos, rope_sin)
        x = x + self.ff(self.prenorm2(x))
        return x

# ------------------- # LearnFormer # ------------------- #
@dataclass
class LearnFormerArgs:
    vocab_size: int = 100277
    dim: int = 4
    nheads: int = 2
    nlayers: int = 2
    cntx: int = 4
    bsz: int = 2
    multiplier: int = 4
    dropout_ff: float = 0.0
    use_mask: bool = True

# Adapted from Easy attention: A simple self-attention mechanism for Transformers
# Essentially, we use the attention score as a learnable parameter
# https://arxiv.org/pdf/2308.12874.pdf 
class LearnableAttentionScore(torch.nn.Module):
    def __init__(self, args: LearnFormerArgs):
        super().__init__()
        self.use_mask = args.use_mask 
        self.alpha = torch.nn.Parameter(torch.rand(args.cntx, args.cntx)) # Attn Score param
        self.register_buffer("mask", torch.tril(torch.ones(args.cntx, args.cntx, requires_grad=False)))

    def forward(self, vi):
        att = self.alpha * self.mask if self.use_mask else self.alpha
        return att @ vi 
    
class MultiEasyAttn(torch.nn.Module):
    def __init__(self, args: LearnFormerArgs):
        super().__init__()
        self.args = args
        self.v_proj = torch.nn.Linear(args.dim, args.dim, bias=False)
        self.alphas = torch.nn.ModuleDict({f'alpha_{i}': LearnableAttentionScore(self.args) for i in range(args.nheads)})
        self.nheads = args.nheads

    def forward(self, x):
        d = x.size(-1)
        v  = self.v_proj(x)
        v  = v.split(d // self.nheads, dim=-1)

        return torch.cat([self.alphas[f'alpha_{i}'](v[i]) for i in range(self.nheads)], dim=-1)


class LearnFormerBlock(torch.nn.Module):
    def __init__(self, args: LearnFormerArgs):
        super().__init__()
        self.attn = MultiEasyAttn(args)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(args.dim, args.dim * args.multiplier),
            torch.nn.GELU(),
            torch.nn.Linear(args.dim * args.multiplier, args.dim),
            torch.nn.Dropout(args.dropout_ff)
        )
        self.ln1 = torch.nn.LayerNorm(args.dim)
        self.ln2 = torch.nn.LayerNorm(args.dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -------------------- # ConvoFormer # -------------------- #