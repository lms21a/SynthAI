import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.nn.functional as F #type: ignore
from einops import rearrange #type: ignore
from einops.layers.torch import Rearrange #type: ignore

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

# ------------------- # FlowFormer  # ------------------- #

# New Attention (supports GQA/MQA but only when flash attention 2 comes out)
class Attention(nn.Module):
    def __init__(self, n_embed, n_heads, n_groups):
        super().__init__()
        assert(n_embed % n_heads == 0), 'Error 811: Embedding Dimension needs to be Divisible by Number of Heads'
        assert(n_embed % n_groups == 0), 'Error 812: Embedding Dimension needs to be Divisible by Number of Groups'
        assert(n_heads == n_groups), 'Error 849: Number of Heads needs to Equal Number of Groups until Flash Attention 2 is implemented'
        self.n_embed = n_embed
        self.qkv = nn.Linear(n_embed, 3*n_embed)
        self.rarr_q = Rearrange('b t (nh hs) -> b nh t hs', nh = n_heads)
        self.rarr_k_v = Rearrange('b t (gs hs) -> b gs t hs', gs = n_groups)
        self.rarr_output = Rearrange('b h t hs -> b t (h hs)', h = n_heads) # This may change depending on what is needed for flash attention 2
        self.fc_out = nn.Linear(n_embed, n_embed)
    
    def forward(self, x):
        q,k,v = self.qkv(x).split(self.n_embed, dim = -1)
        q = self.rarr_q(q)
        k = self.rarr_k_v(k)
        v = self.rarr_k_v(v)
        output = F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=True)
        output = self.rarr_output(output)

        return self.fc_out(output)

# Similar to Llama2
class SwiGLU_FF(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.fc1 = nn.Linear(n_embed, 4*n_embed)
        self.fc2 = nn.Linear(n_embed, 4*n_embed)
        self.output_layer = nn.Linear(4 * n_embed, n_embed)

    def forward(self, x):
        return self.output_layer(F.silu(self.fc1(x) * self.fc2(x)))

# Primer from LabML
def squared_relu(x):
    return F.relu(x)**2
class Squared_FF(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.fc1 = nn.Linear(n_embed, 4*n_embed)
        self.fc2 = nn.Linear(4*n_embed, n_embed)

    def forward(self, x):
        return squared_relu(self.fc2(self.fc1(x)))

# New Transformer Block
class FlowFormer_Block_GLU(nn.Module):
    def __init__(self, n_embed, n_heads, n_groups):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = Attention(n_embed, n_heads, n_groups)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = SwiGLU_FF(n_embed)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class FlowFormer_Block_Squared(nn.Module):
    def __init__(self, n_embed, n_heads, n_groups):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embed)
        self.attn = Attention(n_embed, n_heads, n_groups)
        self.ln2 = nn.LayerNorm(n_embed)
        self.ff = Squared_FF(n_embed)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x