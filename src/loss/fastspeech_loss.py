import torch.nn as nn


class FastSpeechLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, mel_generated, duration_predicted, mel_target, duration_target, **batch):
        mel_loss = self.mse_loss(mel_generated, mel_target)

        duration_loss = self.l1_loss(duration_predicted, duration_target.float())

        return mel_loss + duration_loss
