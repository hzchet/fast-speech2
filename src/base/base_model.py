import torch.nn as nn


class BaseModel(nn.Module):
    def __int__(self, *args, **kwargs):
        super().__init__()
        
    def forward(self, *args, **kwargs):
        raise NotImplementedError()
