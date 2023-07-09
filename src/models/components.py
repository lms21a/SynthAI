import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
from einops import rearrange #type: ignore


class Wte_Wpe(nn.Module):
    def __init__(self, vocab_size, d_model, cntx_len,dropout=0.0):
        super(Wte_Wpe, self).__init__()
        self.d_model = d_model
        self.cntx_len = cntx_len
        self.vocab_size = vocab_size
        self.dropout_p = dropout
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(cntx_len, d_model)

    def forward(self, x):
        pos = torch.arange(0, self.cntx_len,dtype=torch.long, device=x.device)
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

    def forward(self, x):
        x = x + self.csa(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


