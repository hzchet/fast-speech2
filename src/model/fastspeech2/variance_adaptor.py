from typing import Dict

import torch
import torch.nn as nn

from src.model.fastspeech.duration_predictor import BasePredictor
from src.model.fastspeech.length_regulator import LengthRegulator


class PitchPredictor(nn.Module):
    def __init__(
        self,
        max_pitch: float,
        pitch_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float
    ):
        super().__init__()
        self.net = BasePredictor(embed_dim, hidden_dim, kernel_size, dropout)
        self.max_pitch = torch.log1p(torch.tensor(max_pitch))
        self.linspace = torch.linspace(0, 1, pitch_vocab_size)
        self.embedding = nn.Embedding(pitch_vocab_size, embed_dim)
    
    def forward(self, x, target_pitch=None, beta: float = 1.0):
        pred_pitch = self.net(x)
        
        if target_pitch is None:
            log_pitch = torch.log1p(
                (torch.exp(pred_pitch * self.max_pitch) - 1) * beta
            ) / self.max_pitch
        else:
            log_pitch = torch.log1p(target_pitch) / self.max_pitch
        
        log_pitch = torch.clamp(log_pitch, 0, 1)
        pitch_embeds = self.embedding(torch.bucketize(
            log_pitch, self.linspace.to(x.device)
        ))

        return x + pitch_embeds, pred_pitch


class EnergyPredictor(nn.Module):
    def __init__(
        self,
        max_energy: float,
        energy_vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        kernel_size: int,
        dropout: float,
    ):
        super().__init__()
        self.net = BasePredictor(embed_dim, hidden_dim, kernel_size, dropout)
        self.max_energy = max_energy
        self.linspace = torch.linspace(0, 1, energy_vocab_size)
        self.embedding = nn.Embedding(energy_vocab_size, embed_dim)
        
    def forward(self, x, target_energy=None, gamma: float = 1.0):
        pred_energy = self.net(x)

        if target_energy is None:
            energy = pred_energy * gamma
        else:
            energy = target_energy / self.max_energy
        
        energy = torch.clamp(energy, 0, 1)
        energy_embeds = self.embedding(torch.bucketize(
            energy, self.linspace.to(x.device)
        ))
        return energy_embeds, pred_energy


class VarianceAdaptor(nn.Module):
    def __init__(
        self, 
        energy_vocab_size, 
        pitch_vocab_size, 
        max_duration,
        max_energy,
        max_pitch,
        predictor_args: Dict
    ):
        super().__init__()
        self.length_regulator = LengthRegulator(max_duration, predictor_args)
        self.pitch_predictor = PitchPredictor(max_pitch, pitch_vocab_size, **predictor_args)
        self.energy_predictor = EnergyPredictor(max_energy, energy_vocab_size, **predictor_args)
        
    def forward(
        self,
        x,
        duration_target=None,
        mel_max_len=None,
        alpha: float = 1.0,
        pitch_target=None,
        beta: float = 1.0,
        energy_target=None,
        gamma: float = 1.0
    ):
        if self.training:
            regulated, duration_predicted = self.length_regulator(x, duration_target, alpha, 
                                                                  mel_max_len)
        else:
            regulated, duration_predicted, mel_pos = self.length_regulator(x, alpha=alpha)

        out_pitch, pitch_predicted = self.pitch_predictor(regulated, pitch_target, beta)
        out_energy, energy_predicted = self.energy_predictor(regulated, energy_target, gamma)
        
        out = regulated + out_pitch + out_energy
        
        if self.training:
            return out, duration_predicted, pitch_predicted, energy_predicted
        
        return out, duration_predicted, pitch_predicted, energy_predicted, mel_pos
