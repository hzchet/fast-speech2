import argparse
import json
import os
from pathlib import Path

import torch
from tqdm import tqdm

import src.model as module_model
from src.trainer import Trainer
from src.utils import ROOT_PATH, get_WaveGlow
from src.utils.data_loading import get_dataloaders
from src.utils.parse_config import ConfigParser
from src.vocoder.waveglow.inference import inference


DEFAULT_CHECKPOINT_PATH = ROOT_PATH / "default_test_model" / "checkpoint.pth"


def main(config, out_dir):
    logger = config.get_logger("test")

    # define cpu or gpu if possible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # initialize waveglow (vocoder) model
    waveglow_ckpt = config["waveglow_ckpt"]
    WaveGlow = get_WaveGlow(waveglow_ckpt)
    WaveGlow = WaveGlow.to(device)
    
    # setup data_loader instances
    dataloaders, _ = get_dataloaders(config)

    # build model architecture
    model = config.init_obj(config["arch"], module_model)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.to(device)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    with torch.no_grad():
        key = 'test'
        logger.info(f'{key}:')
        
        for batches in tqdm(dataloaders[key]):
            for batch in batches:
                batch = Trainer.move_batch_to_device(batch, device, is_train=False)
                output = model(**batch)
                if type(output) is dict:
                    batch.update(output)
                else:
                    batch["logits"] = output
                
                mel_generated = batch["mel_generated"]
                text_id = batch["text_id"][0]
                alpha = batch["alpha"][0]
                beta = batch["beta"][0]
                gamma = batch["gamma"][0]
                
                inference(
                    mel_generated.contiguous().transpose(1, 2), 
                    WaveGlow, 
                    f'{out_dir}/id={text_id};speed={alpha};pitch={beta};energy={gamma}.wav'
                )


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=str(DEFAULT_CHECKPOINT_PATH.absolute().resolve()),
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    args.add_argument(
        "-t",
        "--texts",
        default=str(ROOT_PATH / "prompts.txt"),
        type=str,
        help="path to a file containing prompts",
    )
    args.add_argument(
        "-o",
        "--output_dir",
        default=str(ROOT_PATH / "saved"),
        type=str,
        help="Dir to save results",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))
    
    assert os.path.exists(args.texts), f'{args.texts} does not exist'
    with open(args.texts, 'r') as f:
        prompts = f.readlines()
    
    config.config["data"] = {
        "test": {
            "batch_size": 1,
            "batch_expand_size": 1,
            "datasets": [
                {
                    "type": "InferenceDataset",
                    "args": {
                        "prompts": prompts,
                        "alphas": [0.8, 1.0, 1.2],
                        "betas": [0.8, 1.0, 1.2],
                        "gammas": [0.8, 1.0, 1.2],
                        "text_cleaners": ["english_cleaners"]
                    },
                }
            ],
        }
    }

    main(config, args.output_dir)
