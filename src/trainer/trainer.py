import logging
import os
from collections import defaultdict

import PIL
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm
import torchaudio

from src.data_processing.text import sequence_to_text
import src.vocoder.waveglow as waveglow
from src.base import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.utils import inf_loop, MetricTracker, get_WaveGlow


logger = logging.getLogger(__name__)


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            train_metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            batch_expand_size=None
    ):
        super().__init__(model, criterion, train_metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50
        self.batch_expand_size = batch_expand_size
        
        self.train_metrics_tracker = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.train_metrics], writer=self.writer
        )
        
        self.num_accumulation_iters = self.config["trainer"].get("num_accumulation_iters", 1)
        waveglow_ckpt = self.config["waveglow_ckpt"]
        WaveGlow = get_WaveGlow(waveglow_ckpt)
        self.WaveGlow = WaveGlow.to(device)

    @staticmethod
    def move_batch_to_device(batch, device: torch.device, is_train: bool):
        """
        Move all necessary tensors to the HPU
        """
        tensors = ['text', 'src_pos']
        if is_train:
            tensors += ['duration_target', 'pitch_target', 'energy_target', 'mel_pos', 'mel_target']
        for tensor_for_gpu in tensors:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics_tracker.reset()
        self.writer.mode = 'train'
        self.writer.add_scalar("epoch", epoch)
        steps = 0
        for outer_batch_idx, batches in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.len_epoch-1)
        ):
            for batch_idx, batch in enumerate(batches):
                try:
                    batch = self.process_batch(
                        batch,
                        is_train=True,
                        metrics=self.train_metrics_tracker,
                        batch_idx=steps
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e) and self.skip_oom:
                        self.logger.warning("OOM on batch. Skipping batch.")
                        for p in self.model.parameters():
                            if p.grad is not None:
                                del p.grad  # free some memory
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
                self.train_metrics_tracker.update("grad norm", self.get_grad_norm())
                real_batch_idx = batch_idx + outer_batch_idx * self.batch_expand_size
                if real_batch_idx % self.log_step == 0:
                    self.writer.set_step(
                        (epoch - 1) * self.len_epoch * self.batch_expand_size + real_batch_idx
                    )
                    self.logger.debug(
                        "Train Epoch: {} {} Loss: {:.6f}".format(
                            epoch, self._progress(outer_batch_idx), batch["loss"].item()
                        )
                    )
                    self.writer.add_scalar(
                        "learning rate", self.lr_scheduler.get_last_lr()[0]
                    )
                    self._log_predictions(**batch)
                    self._log_scalars(self.train_metrics_tracker)
                    # we don't want to reset train metrics at the start of every epoch
                    # because we are interested in recent train metrics
                    last_train_metrics = self.train_metrics_tracker.result()
                    self.train_metrics_tracker.reset()

            if outer_batch_idx + 1 == self.len_epoch:
                break

        log = last_train_metrics
        for part, dataloader in self.evaluation_dataloaders.items():
            self._evaluation_epoch(epoch, part, dataloader)

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker, batch_idx: int = None):
        batch = self.move_batch_to_device(batch, self.device, is_train)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        if is_train:
            batch["loss"] = self.criterion(**batch) / self.num_accumulation_iters
            batch["loss"].backward()
            self._clip_grad_norm()
            if (batch_idx + 1) % self.num_accumulation_iters == 0 or (batch_idx + 1) == self.len_epoch:
                self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            metrics.update("loss", batch["loss"].item())
            if is_train:
                for met in self.train_metrics:
                    metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.writer.mode = part
        total_dict = defaultdict(list)
        with torch.no_grad():
            for batch_idx, batches in enumerate(
                tqdm(dataloader, desc=f"{part}")
            ):
                for batch in batches:
                    batch = self.process_batch(
                        batch,
                        is_train=False,
                        metrics=None
                    )
                    total_dict["mel_generated_list"].append(batch["mel_generated"].squeeze(0))
                    total_dict["text_list"].append(batch["text"].squeeze(0))
                    total_dict["alpha_list"].append(batch["alpha"][0])
                    total_dict["beta_list"].append(batch["beta"][0])
                    total_dict["gamma_list"].append(batch["gamma"][0])

            self._log_inference(**total_dict)

    def synthesis(self, mel):
        return waveglow.inference.get_wav(
            mel.contiguous().transpose(1, 2), self.WaveGlow
        )
    
    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch - 1
        return base.format(current, total, 100.0 * current / total)

    def _log_inference(
        self,
        text_list,
        mel_generated_list,
        alpha_list,
        beta_list,
        gamma_list
    ):
        rows = []
        for text, mel_generated, alpha, beta, gamma in zip(
            text_list, mel_generated_list, alpha_list, beta_list, gamma_list
        ):
            mel_generated_image = PIL.Image.open(plot_spectrogram_to_buf(
                mel_generated.detach().cpu().transpose(-1, -2)
            ))
            mel_generated_image = self.writer.create_image_entry(ToTensor()(mel_generated_image))
            
            gen_audio, sr = self.synthesis(mel_generated.unsqueeze(0))
            gen_audio = self.writer.create_audio_entry(gen_audio, sr)
            
            rows.append({
                "text": sequence_to_text(list(map(int, text))),
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "generated_mel": mel_generated_image,
                "generated_audio": gen_audio
            })

        df = pd.DataFrame(rows)
        self.writer.add_table('predictions', df)

    def _log_predictions(
            self,
            text,
            mel_generated,
            mel_target,
            examples_to_log: int = 5,
            *args,
            **kwargs,
    ):
        save_path = str(self.config.save_dir / 'audios')
        os.makedirs(save_path, exist_ok=True)
        
        batch_size = text.shape[0]
        random_indices = torch.randperm(batch_size)[:examples_to_log]
        
        log_text = text[random_indices].detach().cpu()
        log_mel_generated = mel_generated[random_indices]
        log_mel_target = mel_target[random_indices]
        rows = []
        for i, (example_text, example_mel_generated, example_mel_target) in enumerate(
            zip(log_text, log_mel_generated, log_mel_target)
        ):
            mel_generated_image = PIL.Image.open(plot_spectrogram_to_buf(
                example_mel_generated.detach().cpu().transpose(-1, -2)
            ))
            mel_generated_image = self.writer.create_image_entry(ToTensor()(mel_generated_image))
            
            mel_target_image = PIL.Image.open(plot_spectrogram_to_buf(
                example_mel_target.detach().cpu().transpose(-1, -2)
            ))
            mel_target_image = self.writer.create_image_entry(ToTensor()(mel_target_image))
            
            gen_audio, sr = self.synthesis(example_mel_generated.unsqueeze(0))
            gen_audio = self.writer.create_audio_entry(gen_audio, sr)
            
            target_audio, sr = self.synthesis(example_mel_target.unsqueeze(0))
            target_audio = self.writer.create_audio_entry(target_audio, sr)
            
            rows.append({
                "text": sequence_to_text(list(map(int, example_text))),
                "target_mel": mel_target_image,
                "target_audio": target_audio,
                "generated_mel": mel_generated_image,
                "generated_audio": gen_audio
            })
        
        
        df = pd.DataFrame(rows)
        self.writer.add_table('predictions', df)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
