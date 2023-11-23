import time
import os

from tqdm import tqdm
import numpy as np
import torch

from src.data_processing.text import text_to_sequence
from src.utils import process_text


def get_data_to_buffer(data_path, mel_gt_path, alignment_path, energy_path, pitch_path, 
                       text_cleaners):
    buffer = list()
    text = process_text(data_path)

    start = time.perf_counter()
    for i in tqdm(range(len(text))):
        mel_gt_name = os.path.join(mel_gt_path, "ljspeech-mel-%05d.npy" % (i+1))
        mel_gt_target = np.load(mel_gt_name)
        duration = np.load(os.path.join(alignment_path, str(i)+".npy"))
        energy = np.load(os.path.join(energy_path, str(i)+".npy"))
        pitch = np.load(os.path.join(pitch_path, str(i)+".npy"))
        character = text[i][0:len(text[i])-1]
        character = np.array(text_to_sequence(character, text_cleaners))

        character = torch.from_numpy(character)
        duration = torch.from_numpy(duration)
        energy = torch.from_numpy(energy)
        pitch = torch.from_numpy(pitch)
        mel_gt_target = torch.from_numpy(mel_gt_target)

        buffer.append({
            "text": character, 
            "duration": duration,
            "energy": energy,
            "pitch": pitch,
            "mel_target": mel_gt_target
        })

    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end-start))

    return buffer
