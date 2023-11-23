from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

import src.datasets
from src.collate_fn.collate import collate_fn
from src.utils.parse_config import ConfigParser


def get_dataloaders(configs: ConfigParser):
    dataloaders = {}
    train_batch_expand_size = None
    for split, params in configs["data"].items():
        num_workers = params.get("num_workers", 1)

        # set train augmentations
        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(configs.init_obj(ds, src.datasets))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        # select batch size or batch sampler
        assert "batch_size" in params, "You must provide batch_size for each split"
        assert "batch_expand_size" in params, "You must provide batch_expand_size for each split"

        if split == 'train':
            train_batch_expand_size = params["batch_expand_size"]
        
        if "batch_size" in params:
            bs = params["batch_size"] * params["batch_expand_size"]
            shuffle = True
            batch_sampler = None
        else:
            raise Exception()

        collator = collate_fn(params["batch_expand_size"], split)
        
        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"
        
        pin_memory = params.get("pin_memory", False)
        
        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=collator,
            shuffle=shuffle, num_workers=num_workers,
            batch_sampler=batch_sampler, drop_last=drop_last, pin_memory=pin_memory
        )
        dataloaders[split] = dataloader

    return dataloaders, train_batch_expand_size
