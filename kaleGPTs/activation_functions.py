import torch
import torch.nn as nn

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
