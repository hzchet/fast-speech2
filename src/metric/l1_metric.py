import torch
import torch.nn as nn

from src.base.base_metric import BaseMetric


class L1Metric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = nn.L1Loss()
    
    def __call__(self, duration_predicted, duration_target, **batch):
        with torch.inference_mode():
            return self.metric(duration_predicted, duration_target)
