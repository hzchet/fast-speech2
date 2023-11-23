import torch.nn as nn


class PredictorBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float
    ):
        super().__init__()
        self.conv = nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=1)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        output = self.conv(x.transpose(-1, -2))
        output = self.layer_norm(output.transpose(-1, -2))
        
        return self.dropout(self.relu(output))


class BasePredictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float
    ):
        super().__init__()
        self.net = nn.Sequential(
            PredictorBlock(embed_dim, hidden_dim, kernel_size, dropout),
            PredictorBlock(hidden_dim, hidden_dim, kernel_size, dropout),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DurationPredictor(BasePredictor):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float
    ):
        super().__init__(embed_dim, hidden_dim, kernel_size, dropout)
