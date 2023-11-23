import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        """
        Inputs
            embed_dim - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        super().__init__()
        
        pe = torch.zeros(1, max_len, embed_dim)
        
        numerator = torch.arange(max_len, dtype=torch.float).view(-1, 1)
        reciprocal_denominator = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        grid = numerator * reciprocal_denominator
        
        pe[:, :, 0::2] = torch.sin(grid)
        pe[:, :, 1::2] = torch.cos(grid)

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        x = self.pe[:, :x.size(1), :] + x
        return x
