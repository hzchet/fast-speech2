from typing import List, Dict
import torch.nn as nn


class ConvFFN(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        feed_forward_dim: int,
        kernel_sizes: List[int],
        paddings: List[int],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(embed_dim, feed_forward_dim, kernel_size=kernel_sizes[0], padding=paddings[0]),
            nn.ReLU(),
            nn.Conv1d(feed_forward_dim, embed_dim, kernel_size=kernel_sizes[1], padding=paddings[1])
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        residual = x
        output = self.dropout(self.net(x.transpose(-1, -2)))
        output_norm = self.layer_norm(output.transpose(-1, -2) + residual)
        return output_norm


class FFTBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        dropout: float,
        feed_forward_args: Dict
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, dropout, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.feed_forward = ConvFFN(embed_dim=embed_dim, dropout=dropout, **feed_forward_args)
        
    def forward(self, x, padding_mask=None):
        residual = x

        output, _ = self.attention(x, x, x, attn_mask=padding_mask)
        x_norm = self.layer_norm(residual + output)
        
        return self.feed_forward(x_norm)
