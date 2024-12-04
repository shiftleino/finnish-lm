# KaleGPT2 Differs from KaleGPT by using ALiBi for the positional information

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import activation_functions as actfunc
import regularization as reg
import normalization as norms


class SwiGLUMLP(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.act = actfunc.SwiGLU(self.model_dim)
        self.W_down = nn.Linear(4*self.model_dim, self.model_dim)
        self.hidden_dropout = reg.Dropout()
        self.out_dropout = reg.Dropout()
    
    def forward(self, x: torch.Tensor):
        h = self.act(x)
        h = self.hidden_dropout(h)
        outputs = self.W_down(h)
        outputs = self.out_dropout(outputs)
        return outputs

class MLP(nn.Module):
    def __init__(self, model_dim, act_name):
        super().__init__()
        self.model_dim = model_dim
        self.act_name = act_name
        self.W_up = nn.Linear(self.model_dim, 4*self.model_dim)
        if self.act_name == "relu":
            self.act = actfunc.ReLU()
        elif self.act_name == "lrelu":
            self.act = actfunc.LeakyReLU()
        elif self.act_name == "gelu":
            self.act = actfunc.GELU()
        else:
            raise ValueError(f"Activation function '{self.act}' is not implemented.")
        self.W_down = nn.Linear(4*self.model_dim, self.model_dim)
        self.hidden_dropout = reg.Dropout()
        self.out_dropout = reg.Dropout()
    
    def forward(self, x: torch.Tensor):
        h = self.act(self.W_up(x))
        h = self.hidden_dropout(h)
        outputs = self.W_down(h)
        outputs = self.out_dropout(outputs)
        return outputs

class MultiHeadAttentionALiBi(nn.Module):
    def __init__(self, model_dim, num_heads):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn_dim = self.model_dim // self.num_heads
        
        # All weights grouped together for efficiency as same dim in q, k, v
        self.W_qkv = torch.nn.Linear(self.model_dim, 3*self.model_dim) # 3 for query, key, value
        self.W_o = nn.Linear(self.model_dim, self.model_dim)
        self.attn_dropout = reg.Dropout()
        self.out_dropout = reg.Dropout()

    def forward(self, x: torch.Tensor, alibi: torch.Tensor):
        batch_size, block_size, _ = x.shape
        qkv = self.W_qkv(x)
        qkv = qkv.view(batch_size, block_size, self.num_heads, 3, self.attn_dim) # Split the result between attn_heads
        queries, keys, values = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :] # non-contiguous in memory!!
        q = queries.transpose(1, 2) # (batch, block, head, dim) -> (batch, head, block, dim)
        k = keys.transpose(1,2) # (batch, block, head, dim) -> (batch, head, block, dim)
        qk = q @ k.transpose(2, 3) / math.sqrt(self.attn_dim)
        masked_qk = qk + alibi
        attn_o = F.softmax(masked_qk, dim=-1) @ values.transpose(1, 2)
        attn_o = self.attn_dropout(attn_o)
        outputs = self.W_o(attn_o.transpose(1, 2).reshape(batch_size, block_size, self.model_dim)) # Use reshape as tensor is not contiguous due to transposes
        outputs = self.out_dropout(outputs)
        return outputs

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, act: str, norm: str):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.act = act
        self.norm = norm

        if norm == "layer":
            self.norm_pre_attn = norms.LayerNorm(self.model_dim, 1e-5)
            self.norm_pre_mlp = norms.LayerNorm(self.model_dim, 1e-5)
        elif norm == "rms":
            self.norm_pre_attn = norms.RMSNorm(self.model_dim)
            self.norm_pre_mlp = norms.RMSNorm(self.model_dim)
        else:
            raise ValueError(f"Undefined normalization: {self.norm}")
        
        self.attn = MultiHeadAttentionALiBi(self.model_dim, self.num_heads)
        if self.act == "swiglu":
            self.mlp = SwiGLUMLP(self.model_dim)
        else:
            self.mlp = MLP(self.model_dim, self.act)

    def forward(self, x: torch.Tensor, alibi: torch.Tensor):
        attn_output = self.attn(self.norm_pre_attn(x), alibi) + x
        mlp_output = self.mlp(self.norm_pre_mlp(attn_output)) + attn_output
        return mlp_output

class KaleGPT2(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, max_block_size, act, norm, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_block_size = max_block_size
        self.act = act
        self.norm = norm
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.transformer_blocks = nn.ModuleList(TransformerBlock(self.model_dim, self.num_heads, self.act, self.norm) for _ in range(self.num_layers))
        if norm == "layer":
            self.final_norm = norms.LayerNorm(self.model_dim, 1e-5)
        elif norm == "rms":
            self.final_norm = norms.RMSNorm(self.model_dim)
        else:
            raise ValueError(f"Undefined normalization: {self.norm}")
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size)
    
    def forward(self, x: torch.Tensor):
        block_size = min(x.shape[1], self.max_block_size) # Allow for max_block_size tokens, can differ from training block_size

        representations = self.embedding(x[:, -block_size:])

        mask = torch.tril(torch.ones((block_size, block_size))).to(self.device)
        bias = torch.arange(block_size).unsqueeze(0).to(self.device) - torch.arange(block_size).unsqueeze(1).to(self.device)
        slopes = torch.pow(2.0, -8/self.num_heads * torch.arange(1, self.num_heads + 1).to(self.device))
        alibi = slopes.view(self.num_heads, 1, 1) * bias.view(1, block_size, block_size).unsqueeze(0)
        alibi = torch.masked_fill(alibi, mask == 0, float("-inf"))

        for transformer_block in self.transformer_blocks:
            representations = transformer_block(representations, alibi)
        representations = self.final_norm(representations)
        logits = self.lm_head(representations)
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
        return f"KaleGPT2(vocab_size={self.vocab_size}, model_dim={self.model_dim}, num_heads={self.num_heads}, num_layers={self.num_layers})"
