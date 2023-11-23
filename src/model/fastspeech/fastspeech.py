from typing import Dict

import torch
import torch.nn as nn

from src.utils import get_padding_mask, get_mask_from_lengths
from src.base.base_model import BaseModel
from src.model.fastspeech.pos_encoder import PositionalEncoding
from src.model.fastspeech.fft_block import FFTBlock
from src.model.fastspeech.length_regulator import LengthRegulator


class FastSpeechEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: int,
        max_seq_len: int,
        n_heads: int,
        dropout: float,
        feed_forward_args: Dict,
        n_blocks: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.padding_idx = padding_idx
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.pos_encoder = nn.Embedding(max_seq_len + 1, embed_dim, padding_idx=padding_idx)
        blocks = [
            FFTBlock(embed_dim, n_heads, dropout, feed_forward_args) for _ in range(n_blocks)
        ]
        self.encoder = nn.ModuleList(blocks)
        
    def forward(self, x, src_pos):
        padding_mask = get_padding_mask(x, x, self.padding_idx)
        padding_mask = padding_mask.repeat(self.n_heads, 1, 1)
        
        output = self.embedding(x) + self.pos_encoder(src_pos)
        
        for fft_block in self.encoder:
            output = fft_block(output, padding_mask)
        
        return output


class FastSpeechDecoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        padding_idx: int,
        max_seq_len: int,
        n_heads: int,
        dropout: float,
        feed_forward_args: Dict,
        n_blocks: int
    ):
        super().__init__()
        self.n_heads = n_heads
        self.padding_idx = padding_idx
        self.pos_encoder = nn.Embedding(max_seq_len + 1, embed_dim, padding_idx=padding_idx)
        blocks = [
            FFTBlock(embed_dim, n_heads, dropout, feed_forward_args) for _ in range(n_blocks)
        ]
        self.decoder = nn.ModuleList(blocks)
        
    def forward(self, x, mel_pos):
        padding_mask = get_padding_mask(mel_pos, mel_pos, self.padding_idx)
        padding_mask = padding_mask.repeat(self.n_heads, 1, 1)
        output = x + self.pos_encoder(mel_pos)
        
        for fft_block in self.decoder:
            output = fft_block(output, padding_mask)
            
        return output


class FastSpeech(BaseModel):
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
        duration_predictor_args: Dict,
        num_mels: int
    ):
        super().__init__()
        self.encoder = FastSpeechEncoder(vocab_size, embed_dim, padding_idx, max_seq_len, n_heads,
                                         dropout, feed_forward_args, n_blocks)
        duration_predictor_args.update({"embed_dim": embed_dim, "dropout": dropout})
        self.length_regulator = LengthRegulator(duration_predictor_args)
        
        

    def forward(
        self, 
        text, 
        src_pos, 
        duration_target=None,
        mel_pos=None, 
        mel_max_len=None, 
        alpha: float = 1.0,
        **batch
    ):
        encoder_output = self.encoder(text, src_pos)
        
        if self.training:
            regulated_length, duration_predicted = self.length_regulator(encoder_output,
                                                                         duration_target,
                                                                         alpha, mel_max_len)
            decoder_output = self.decoder(regulated_length, mel_pos)
            masked = self.mask_tensor(decoder_output, mel_pos, mel_max_len)
            
            mel_predicted = self.mel_linear(masked)
            return {
                "mel_generated": mel_predicted,
                "duration_predicted": duration_predicted
            }
        
        regulated_length, duration_predicted, mel_pos = self.length_regulator(encoder_output,
                                                                              alpha=alpha)
        decoder_output = self.decoder(regulated_length, mel_pos)
        mel_predicted = self.mel_linear(decoder_output)
        
        return {
            "mel_generated": mel_predicted,
            "duration_predicted": duration_predicted
        }

    def mask_tensor(self, decoder_output, mel_pos, mel_max_len):
        lengths = torch.max(mel_pos, -1)[0]
        mask = ~get_mask_from_lengths(lengths, max_len=mel_max_len)
        mask = mask.unsqueeze(-1).expand(-1, -1, decoder_output.size(-1))
        
        return decoder_output.masked_fill(mask, 0.)
