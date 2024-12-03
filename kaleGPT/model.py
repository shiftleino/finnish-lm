import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReLU(nn.Module):
    def forward(self, x):
        x = torch.clamp(x, min=0)
        return x
    
class LeakyReLU(nn.Module):
    def forward(self, x):
        x = torch.where(x >= 0, x, 0.01*x)
        return x

class GELU(nn.Module):
    def forward(self, x):
        x = 0.5*x*(1.0 + torch.tanh(0.7978845608*(x+0.044715*x*x*x))) # approximation from Hendrycks & Gimpel (2016)
        return x

class SiLU(nn.Module):
    def forward(self, x):
        x = x * 1/(1+torch.exp(-x))
        return x

class SwiGLU(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.W_gate = nn.Linear(self.model_dim, 4*self.model_dim)
        self.W_up = nn.Linear(self.model_dim, 4*self.model_dim)
        self.swish = SiLU()

    def forward(self, x):
        gate = self.swish(self.W_gate(x))
        outputs = gate * self.W_up(x)
        return outputs

class Dropout(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super().__init__()
        self.dropout_rate = dropout_rate

    def forward(self, x):
        if self.training:
            mask = (torch.rand_like(x) < (1-self.dropout_rate)).float()
            return x * mask * (1 / (1 - self.dropout_rate)) # apply scaling in training different from original article (Srivastava et al., 2014)
        else:
            return x

class SwiGLUMLP(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.act = SwiGLU(self.model_dim)
        self.W_down = nn.Linear(4*self.model_dim, self.model_dim)
        self.hidden_dropout = Dropout()
        self.out_dropout = Dropout()
    
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
            self.act = ReLU()
        elif self.act_name == "lrelu":
            self.act = LeakyReLU()
        elif self.act_name == "gelu":
            self.act = GELU()
        else:
            raise ValueError(f"Activation function '{self.act}' is not implemented.")
        self.W_down = nn.Linear(4*self.model_dim, self.model_dim)
        self.hidden_dropout = Dropout()
        self.out_dropout = Dropout()
    
    def forward(self, x: torch.Tensor):
        h = self.act(self.W_up(x))
        h = self.hidden_dropout(h)
        outputs = self.W_down(h)
        outputs = self.out_dropout(outputs)
        return outputs

class MultiHeadAttention(nn.Module):
    def __init__(self, model_dim, num_heads, max_block_size):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.attn_dim = self.model_dim // self.num_heads
        self.max_block_size = max_block_size
        
        # All weights grouped together for efficiency as same dim in q, k, v
        self.W_qkv = torch.nn.Linear(self.model_dim, 3*self.model_dim) # 3 for query, key, value
        self.W_o = nn.Linear(self.model_dim, self.model_dim)
        self.attn_dropout = Dropout()
        self.out_dropout = Dropout()
        self.register_buffer("mask", torch.tril(torch.ones((self.max_block_size, self.max_block_size))) == 0)

    def forward(self, x: torch.Tensor):
        batch_size, block_size, _ = x.shape
        qkv = self.W_qkv(x)
        qkv = qkv.view(batch_size, block_size, self.num_heads, 3, self.attn_dim) # Split the result between attn_heads
        queries, keys, values = qkv[:, :, :, 0, :], qkv[:, :, :, 1, :], qkv[:, :, :, 2, :] # non-contiguous in memory!!
        q = queries.transpose(1, 2) # (batch, block, head, dim) -> (batch, head, block, dim)
        k = keys.transpose(1,2) # (batch, block, head, dim) -> (batch, head, block, dim)
        qk = q @ k.transpose(2, 3) / math.sqrt(self.attn_dim)
        masked_qk = torch.masked_fill(qk, self.mask[:block_size, :block_size], float("-inf"))
        attn_o = F.softmax(masked_qk, dim=-1) @ values.transpose(1, 2)
        attn_o = self.attn_dropout(attn_o)
        outputs = self.W_o(attn_o.transpose(1, 2).reshape(batch_size, block_size, self.model_dim)) # Use reshape as tensor is not contiguous due to transposes
        outputs = self.out_dropout(outputs)
        return outputs

class LayerNorm(nn.Module):
    def __init__(self, model_dim, eps):
        super().__init__()
        self.eps = eps
        self.model_dim = model_dim
        self.gamma = nn.Parameter(torch.ones(self.model_dim))
        self.beta = nn.Parameter(torch.zeros(self.model_dim))
    
    def forward(self, x):
        normalized = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        return self.gamma * normalized + self.beta

class RMSNorm(nn.Module):
    def __init__(self, model_dim, eps=1e-5):
        super().__init__()
        self.model_dim = model_dim
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(self.model_dim))

    def forward(self, x):
        rms = ((x**2).sum(dim=-1, keepdim=True) / x.size(-1))**(0.5)
        normalized = x / (rms + self.eps)
        return self.gain * normalized

class TransformerBlock(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, max_block_size: int, act: str, norm: str):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.max_block_size = max_block_size
        self.act = act
        self.norm = norm

        if norm == "layer":
            self.norm_pre_attn = LayerNorm(self.model_dim, 1e-5)
            self.norm_pre_mlp = LayerNorm(self.model_dim, 1e-5)
        elif norm == "rms":
            self.norm_pre_attn = RMSNorm(self.model_dim)
            self.norm_pre_mlp = RMSNorm(self.model_dim)
        else:
            raise ValueError(f"Undefined normalization: {self.norm}")
        
        self.attn = MultiHeadAttention(self.model_dim, self.num_heads, self.max_block_size)
        if self.act == "swiglu":
            self.mlp = SwiGLUMLP(self.model_dim)
        else:
            self.mlp = MLP(self.model_dim, self.act)

    def forward(self, x):
        attn_output = self.attn(self.norm_pre_attn(x)) + x
        mlp_output = self.mlp(self.norm_pre_mlp(attn_output)) + attn_output
        return mlp_output

class KaleGPT(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_layers, block_size, act, norm, device):
        super().__init__()
        self.vocab_size = vocab_size
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.block_size = block_size
        self.act = act
        self.norm = norm
        self.device = device
        self.embedding = nn.Embedding(self.vocab_size, self.model_dim)
        self.pos_embedding = nn.Embedding(self.block_size, self.model_dim)
        self.transformer_blocks = nn.ModuleList(TransformerBlock(self.model_dim, self.num_heads, self.block_size, self.act, self.norm) for _ in range(self.num_layers))
        if norm == "layer":
            self.final_norm = LayerNorm(self.model_dim, 1e-5)
        elif norm == "rms":
            self.final_norm = RMSNorm(self.model_dim)
        else:
            raise ValueError(f"Undefined normalization: {self.norm}")
        self.lm_head = nn.Linear(self.model_dim, self.vocab_size)
    
    def forward(self, x: torch.Tensor):
        word_embeddings = self.embedding(x)
        positions = torch.arange(x.shape[1], device=self.device)
        pos_embeddings = self.pos_embedding(positions)
        representations = word_embeddings + pos_embeddings
        for transformer_block in self.transformer_blocks:
            representations = transformer_block(representations)
        representations = self.final_norm(representations)
        logits = self.lm_head(representations)
        return logits

    def generate(self, x, max_tokens: int):
        for _ in range(max_tokens):
            logits = self.forward(x[:, -self.block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_idx = torch.multinomial(probs, num_samples=1)
            x = torch.cat([x, next_idx], dim=-1)
        return x

    def __repr__(self):
        return f"KaleGPT(vocab_size={self.vocab_size}, model_dim={self.model_dim}, num_heads={self.num_heads}, num_layers={self.num_layers})"
