import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GPT-Reg import TransformerBlock

class LayerNorm_nb(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = torch.nn.Parameter(torch.ones(features))
        self.beta = None
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        x = (x - mean) / (std + self.eps)
        return self.gamma * x




class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        matmul_qk = torch.matmul(query, key.transpose(-1, -2))

        # Scale matmul_qk
        dk = torch.tensor(key.size(-1), dtype=torch.float32)
        scaled_attention_logits = matmul_qk / torch.sqrt(dk)

        # Apply mask if it exists
        if mask is not None:
            scaled_attention_logits += (mask * -1e9) 

        # Apply softmax to get weights
        attention_weights = F.softmax(scaled_attention_logits, dim=-1)

        # Compute output
        output = torch.matmul(attention_weights, value)

        return output, attention_weights

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        self.dense = torch.nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)

    def forward(self, v, k, q, mask):
        batch_size = q.shape[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = self.attention(q, k, v, mask)

        scaled_attention = scaled_attention.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        output = self.dense(scaled_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class TransformerBlock(torch.nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm_nb(d_model)
        self.norm2 = LayerNorm_nb(d_model)

        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(d_model, 4 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(4 * d_model, d_model),
        )

    def forward(self, value, key, query, mask):
        attention, _ = self.attention(value, key, query, mask)
        x = self.norm1(attention + query)
        ffn_output = self.ffn(x)
        output = self.norm2(ffn_output + x)
        return output


test_transformer_block()