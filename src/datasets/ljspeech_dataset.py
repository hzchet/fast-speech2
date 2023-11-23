import os
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from src.base.base_dataset import BaseDataset
from src.datasets.utils import get_data_to_buffer
from src.data_processing.text import text_to_sequence
from src.utils import process_text


class LJSpeechDataset(BaseDataset):
    def __init__(
        self,
        data_path: str,
        mel_gt_path: str,
        alignment_path: str,
        energy_path: str = None,
        pitch_path: str = None,
        text_cleaners: List[str] = ['english_cleaners'],
        *args,
        **kwargs
    ):
        index = get_data_to_buffer(
            data_path, 
            mel_gt_path, 
            alignment_path,
            energy_path,
            pitch_path,
            text_cleaners
        )
        
        super().__init__(index, *args, **kwargs)


class LJSpeechDatasetUnbuffered(Dataset):
    def __init__(
        self,
        data_path: str,
        mel_gt_path: str,
        alignment_path: str,
        energy_path: str = None,
        pitch_path: str = None,
        text_cleaners: List[str] = ['english_cleaners'],
        limit: int = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.text = process_text(data_path)
        if limit is not None and 0 <= limit < len(self.text):
            self.text = self.text[:limit]

        self.mel_gt_path = mel_gt_path
        self.alignment_path = alignment_path
        self.energy_path = energy_path
        self.pitch_path = pitch_path
        self.text_cleaners = text_cleaners

    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, i: int):
        mel_gt_name = os.path.join(self.mel_gt_path, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(self.alignment_path, str(i)+".npy"))
        energy = np.load(os.path.join(self.energy_path, str(i)+".npy"))
        pitch = np.load(os.path.join(self.pitch_path, str(i)+".npy"))
        character = self.text[i][0:len(self.text[i])-1]
        character = np.array(text_to_sequence(character, self.text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        energy = torch.from_numpy(energy)
        pitch = torch.from_numpy(pitch)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        return {
            "text": character, 
            "duration": duration,
            "energy": energy,
            "pitch": pitch,
            "mel_target": mel_gt_target
        }
