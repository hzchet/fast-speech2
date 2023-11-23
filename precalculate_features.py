import os
import argparse
import warnings

import torchaudio
import numpy as np
from tqdm import tqdm

from src.utils.parse_config import ConfigParser
from src.utils import get_pitch, get_energy

warnings.filterwarnings("ignore")


def main(config: ConfigParser):
    wavs_dir = config["wavs_dir"]
    assert os.path.exists(wavs_dir)
    
    pitches_dir = config["pitches_dir"]
    os.makedirs(pitches_dir, exist_ok=True)
    
    energies_dir = config["energies_dir"]
    os.makedirs(energies_dir, exist_ok=True)
    
    wav_names = sorted(os.listdir(wavs_dir))
    
    min_pitch, max_pitch, min_energy, max_energy = -np.inf, np.inf, -np.inf,  np.inf
    for i, wav_name in enumerate(tqdm(wav_names, "Processing wavs...")):
        audio, _ = torchaudio.load(os.path.join(wavs_dir, wav_name))
        audio = audio.squeeze(0).numpy().astype(np.float64)
        pitch = get_pitch(audio)
        np.save(os.path.join(pitches_dir, f'{i}.npy'), pitch)
        min_pitch = min(min_pitch, pitch.min())
        max_pitch = max(max_pitch, pitch.max())
        
        energy = get_energy(audio)
        np.save(os.path.join(energies_dir, f'{i}.npy'), energy)
        min_energy = min(min_energy, energy.min())
        max_energy = max(max_energy, energy.max())
    
    np.save(os.path.join(pitches_dir, 'minmax.npy'), np.array([min_pitch, max_pitch]))
    np.save(os.path.join(energies_dir, 'minmax.npy'), np.array([min_energy, max_energy]))
    

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    
    config = ConfigParser.from_args(args)
    main(config)
