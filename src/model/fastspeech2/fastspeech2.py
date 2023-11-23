from typing import Dict

import torch
import torch.nn as nn

from src.base.base_model import BaseModel
from src.utils import get_mask_from_lengths
from src.model.fastspeech import FastSpeechEncoder, FastSpeechDecoder
from src.model.fastspeech2.variance_adaptor import VarianceAdaptor


class FastSpeech2(BaseModel):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int,
        max_seq_len: int,
        n_heads: int,
        dropout: float,
        feed_forward_args: Dict,
        n_blocks: int,
        energy_vocab_size: int,
        pitch_vocab_size: int,
        max_duration: int,
        max_energy: int,
        max_pitch: int,
        predictor_args: Dict,
        num_mels: int
    ):
        super().__init__()
        self.encoder = FastSpeechEncoder(vocab_size, embed_dim, padding_idx, max_seq_len, n_heads,
                                         dropout, feed_forward_args, n_blocks)
        
        predictor_args.update({"embed_dim": embed_dim, "dropout": dropout})
        self.variance_adaptor = VarianceAdaptor(energy_vocab_size, pitch_vocab_size, max_duration,
                                                max_energy, max_pitch, predictor_args)
        
        self.decoder = FastSpeechDecoder(embed_dim, padding_idx, max_seq_len, n_heads, dropout,
                                         feed_forward_args, n_blocks)

        self.mel_linear = nn.Linear(embed_dim, num_mels)
        
    def forward(
        self,
        text,
        src_pos,
        duration_target=None,
        mel_pos=None,
        mel_max_len=None,
        alpha: float = 1.0,
        pitch_target=None,
        beta: float = 1.0,
        energy_target=None,
        gamma: float = 1.0,
        **batch
    ):
        encoder_output = self.encoder(text, src_pos)
        
        if self.training:
            adapted, duration_predicted, pitch_predicted, energy_predicted = self.variance_adaptor(
                encoder_output,
                duration_target,
                mel_max_len,
                alpha,
                pitch_target,
                beta,
                energy_target,
                gamma
            )
            decoder_output = self.decoder(adapted, mel_pos)
            masked = self.mask_tensor(decoder_output, mel_pos, mel_max_len)
            mel_predicted = self.mel_linear(masked) 
        else:
            adapted, duration_predicted, pitch_predicted, energy_predicted, mel_pos = self.variance_adaptor(
                encoder_output,
                alpha,
                beta,
                gamma
            )
            decoder_output = self.decoder(adapted, mel_pos)
            mel_predicted = self.mel_linear(decoder_output)
        
        return {
            "mel_generated": mel_predicted,
            "duration_predicted": duration_predicted,
            "pitch_predicted": pitch_predicted,
            "energy_predicted": energy_predicted
        }

    def mask_tensor(self, decoder_output, mel_pos, mel_max_len):
        lengths = torch.max(mel_pos, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_len)
        mask = mask.unsqueeze(-1).expand(-1, -1, decoder_output.size(-1))
        
        return decoder_output.masked_fill(mask, 0.)
