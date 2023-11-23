import torch
import torch.nn as nn


class FastSpeech2Loss(nn.Module):
    def __init__(self, max_pitch, max_energy, max_duration):
        super().__init__()
        self.max_pitch = torch.log1p(torch.tensor(max_pitch))
        self.max_duration = torch.log1p(torch.tensor(max_duration))
        self.max_energy = max_energy
        self.mse_loss = nn.MSELoss()

    def forward(
        self, 
        mel_generated,
        mel_target, 
        duration_predicted,
        duration_target,
        pitch_predicted,
        pitch_target,
        energy_predicted,
        energy_target,
        **batch
    ):
        mel_loss = self.mse_loss(mel_generated, mel_target)
        duration_loss = self.mse_loss(
            duration_predicted,
            torch.log1p(duration_target.float()) / self.max_duration
        )
        pitch_loss = self.mse_loss(
            pitch_predicted,
            torch.log1p(pitch_target.float()) / self.max_pitch
        )
        energy_loss = self.mse_loss(
            energy_predicted,
            energy_target.float() / self.max_energy
        )
        
        return mel_loss + duration_loss + pitch_loss + energy_loss
