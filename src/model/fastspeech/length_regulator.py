from typing import Dict, Union, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.fastspeech.duration_predictor import DurationPredictor


def create_alignment(base_mat, duration_predictor_output):
    N, L = duration_predictor_output.shape
    for i in range(N):
        count = 0
        for j in range(L):
            for k in range(duration_predictor_output[i][j]):
                base_mat[i][count+k][j] = 1
            count = count + duration_predictor_output[i][j]
    return base_mat


class LengthRegulator(nn.Module):
    def __init__(self, max_duration, duration_predictor_args: Dict):
        super().__init__()
        self.max_log_duration = torch.log1p(torch.tensor(max_duration))
        self.duration_predictor = DurationPredictor(**duration_predictor_args)
    
    def regulate(self, x, duration, mel_max_len=None):
        expand_max_len = torch.max(torch.sum(duration, -1), -1)[0]
        alignment = torch.zeros(duration.size(0), expand_max_len, duration.size(1)).numpy()
        alignment = create_alignment(alignment, duration.cpu().numpy())
        alignment = torch.from_numpy(alignment).to(x.device)

        output = alignment @ x
        if mel_max_len:
            output = F.pad(output, (0, 0, 0, mel_max_len - output.size(1), 0, 0))

        return output
    
    def forward(self, x, target_duration=None, alpha: float = 1.0, mel_max_len: int = None):
        pred_duration = self.duration_predictor(x)
        
        if target_duration is not None:
            regulated = self.regulate(x, target_duration, mel_max_len)
            return regulated, pred_duration
        
        pred_duration = ((torch.exp(pred_duration * self.max_log_duration) - 1) * alpha[0] + 0.5).int()
        regulated = self.regulate(x, pred_duration)
        
        mel_pos = list()
        B = x.shape[0]
        for _ in range(B):
            mel_pos.append([i + 1 for i in range(int(regulated.size(1)))])

        mel_pos = torch.from_numpy(np.array(mel_pos)).long().to(x.device)
        
        return regulated, pred_duration, mel_pos
