import random

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(
        self, 
        index,
        limit=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self._index = self._filter_records_from_dataset(index, limit)
    
    def __getitem__(self, ind):
        return self._index[ind]
    
    def __len__(self):
        return len(self._index)

    @staticmethod
    def _filter_records_from_dataset(
            index: list, limit
    ) -> list:
        if limit is not None:
            random.seed(42)  # best seed for deep learning
            random.shuffle(index)
            index = index[:limit]
        return index
