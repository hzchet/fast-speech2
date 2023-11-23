from typing import List

import torch

from src.base.base_dataset import BaseDataset
import src.data_processing.text as text


class InferenceDataset(BaseDataset):
    def __init__(
        self, 
        prompts: List[str], 
        alphas: List[int], 
        betas: List[int],
        gammas: List[int],
        text_cleaners: List[str], 
        *args, 
        **kwargs
    ):
        index = self._create_index(prompts, alphas, betas, gammas, text_cleaners)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, prompts: List[str], alphas: List[int], betas: List[int], gammas: List[int], text_cleaners: List[str]):
        index = []
        for i, prompt in enumerate(prompts):
            for alpha in alphas:
                for beta in betas:
                    for gamma in gammas:
                        index.append({
                            "text_id": i,
                            "text": torch.tensor(text.text_to_sequence(prompt, text_cleaners)),
                            "alpha": alpha,
                            "beta": beta,
                            "gamma": gamma
                        })

        return index
