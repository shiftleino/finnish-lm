import torch
import torch.nn as nn

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
