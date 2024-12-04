import torch
import torch.nn as nn

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
