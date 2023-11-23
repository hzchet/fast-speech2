import numpy as np
import torch

from src.utils import pad_1D_tensor, pad_2D_tensor


def reprocess_tensor(batch, cut_list, split):
    texts = [batch[ind]["text"] for ind in cut_list]

    length_text = np.array([])
    assert texts, print(texts)
    for text in texts:
        length_text = np.append(length_text, text.size(0))

    src_pos = list()
    max_len = int(max(length_text))
    for length_src_row in length_text:
        src_pos.append(np.pad([i+1 for i in range(int(length_src_row))],
                              (0, max_len-int(length_src_row)), 'constant'))
    src_pos = torch.from_numpy(np.array(src_pos))
    texts = pad_1D_tensor(texts)
    if split == 'eval' or split == 'test':
        text_ids = [batch[ind]["text_id"] for ind in cut_list]
        alphas = [batch[ind]["alpha"] for ind in cut_list]
        betas = [batch[ind]["beta"] for ind in cut_list]
        gammas = [batch[ind]["gamma"] for ind in cut_list]
        return {
            "text_id": text_ids,
            "text": texts,
            "src_pos": src_pos,
            "alpha": alphas,
            "beta": betas,
            "gamma": gammas
        }
    
    mel_targets = [batch[ind]["mel_target"] for ind in cut_list]
    durations = [batch[ind]["duration"] for ind in cut_list]
    energies = [batch[ind]["energy"] for ind in cut_list]
    pitches = [batch[ind]["pitch"] for ind in cut_list]
    length_mel = np.array(list())
    for mel in mel_targets:
        length_mel = np.append(length_mel, mel.size(0))

    mel_pos = list()
    max_mel_len = int(max(length_mel))
    for length_mel_row in length_mel:
        mel_pos.append(np.pad([i+1 for i in range(int(length_mel_row))],
                              (0, max_mel_len-int(length_mel_row)), 'constant'))
    mel_pos = torch.from_numpy(np.array(mel_pos))

    durations = pad_1D_tensor(durations)
    energies = pad_1D_tensor(energies)
    pitches = pad_1D_tensor(pitches)
    mel_targets = pad_2D_tensor(mel_targets)

    out = {"text": texts,
           "mel_target": mel_targets,
           "duration_target": durations,
           "energy_target": energies,
           "pitch_target": pitches,
           "mel_pos": mel_pos,
           "src_pos": src_pos,
           "mel_max_len": max_mel_len}

    return out


def collate_fn(batch_expand_size: int, split: str):
    def wrapped(batch):
        len_arr = np.array([d["text"].size(0) for d in batch])
        index_arr = np.argsort(-len_arr)
        batchsize = len(batch)
        real_batchsize = batchsize // batch_expand_size

        cut_list = list()
        for i in range(batch_expand_size):
            cut_list.append(index_arr[i*real_batchsize:(i+1)*real_batchsize])

        output = list()
        for i in range(batch_expand_size):
            output.append(reprocess_tensor(batch, cut_list[i], split))

        return output
    
    return wrapped
