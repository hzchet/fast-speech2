import torch
import torch.nn as nn

from src.base.base_metric import BaseMetric


class MSEMetric(BaseMetric):
    def __init__(
        self, 
        *args, 
        apply_on: str = 'mel',
        max_pitch=None,
        max_energy=None,
        max_duration=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.apply_on = apply_on
        if max_pitch is not None:
            self.max_pitch = torch.log1p(torch.tensor(max_pitch))
        if max_duration is not None:
            self.max_duration = torch.log1p(torch.tensor(max_duration))
        if max_energy is not None:
            self.max_energy = max_energy

        self.metric = nn.MSELoss()
    
    def __call__(
        self,
        mel_generated, 
        mel_target,
        duration_predicted,
        duration_target,
        pitch_predicted=None,
        pitch_target=None,
        energy_predicted=None,
        energy_target=None,
        **batch
    ):
        with torch.inference_mode():
            if self.apply_on == 'mel':
                return self.metric(mel_generated, mel_target)
            if self.apply_on == 'duration':
                return self.metric(
                    duration_predicted,
                    torch.log1p(duration_target.float()) / self.max_duration
                )
            if self.apply_on == 'pitch':
                return self.metric(
                    pitch_predicted,
                    torch.log1p(pitch_target.float()) / self.max_pitch
                )
            if self.apply_on == 'energy':
                return self.metric(
                    energy_predicted,
                    energy_target.float() / self.max_energy
                )
            else:
                raise ValueError()
