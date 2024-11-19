import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor):
        pass

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, block_size):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn_dim = self.model_dim // self.num_heads
        self.block_size = block_size

        # All weights grouped together for efficiency as same dim in q, k, v
        self.W_qkv = torch.nn.Linear(self.model_dim, 3*self.model_dim) # 3 for query, key, value
        self.W_o = nn.Linear(self.model_dim, self.model_dim)
        self.register_buffer("attn_mask", torch.tril(torch.ones(block_size, block_size)) == 0)
    
    def forward(self, x: torch.Tensor):
        batch_size, block_size, _ = x.shape
        qkv = self.W_qkv(x)
        qkv = qkv.view(batch_size, block_size, self.num_heads, 3, self.attn_dim) # Split the result between attn_heads
        queries, keys, values = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :] # non-contiguous in memory!!
        q = queries.transpose(1, 2) # (batch, block, head, dim) -> (batch, head, block, dim)
        k = keys.transpose(1,2) # (batch, block, head, dim) -> (batch, head, block, dim)
        qk = q @ k.transpose(2, 3) / math.sqrt(self.attn_dim)
        masked_qk = torch.masked_fill(qk, self.attn_mask, float("-inf"))
        attn_o = F.softmax(masked_qk, dim=-1) @ values.transpose(1, 2)
        outputs = self.W_o(attn_o.transpose(1, 2).reshape(batch_size, block_size, self.model_dim)) # Use reshape as tensor is not contiguous due to transposes
        return outputs

class KaleGPT(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, block_size, num_layers):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.attn = MultiHeadAttention(self.model_dim, self.num_heads, self.block_size)
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size)
    
    def forward(self, x: torch.Tensor):
        x = self.embedding(x)
        x = self.attn(x) + x
        logits = self.lm_head(x)
        return logits

    def generate(self, x, max_tokens: int):
        for _ in range(max_tokens):
            logits = self.forward(x)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_idx], dim=-1)
        return x

    def __repr__(self):
        return f"KaleGPT(vocab_size={self.vocab_size}, model_dim={self.model_dim}, num_heads={self.num_heads}, num_layers={self.num_layers})"
