"""
src.trainer.py
Base Trainers Classes
BoMeyering 2025
"""

import math
import torch
import os
import json
import time
import wandb
from glob import glob
import uuid
import logging
import argparse
import numpy as np
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import numpy as np

from typing import Union, Optional, Any, Tuple, List
from torchmetrics import MeanMetric
from src.flexmatch import get_pseudo_labels, class_beta
from src.parameters import EMA, apply_ema
# from src.callbacks import ModelCheckpoint
from src.metrics import MetricLogger, MeterSet, RunningAvgMeter, ValueMeter
from src.transforms import get_strong_transforms
from src.distributed import is_main_process
from src.utils.loggers import rank_log
from src.callbacks import CheckpointManager

class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    @abstractmethod
    def _train_step(self, batch) -> Tuple[Any, Any]:
        """Implement the train step for one batch"""
        ...

    @abstractmethod
    def _val_step(self, batch) -> Tuple[Any, Any]:
        """Implement the val step for one batch"""
        ...

    @abstractmethod
    def _train_epoch(self, epoch) -> Any:
        """Implement the training method for one epoch"""
        ...

    @abstractmethod
    def _val_epoch(self, epoch) -> Any:
        """Implement the validation method for one epoch"""
        ...

    @abstractmethod
    def train(self):
        """Implement the whole training loop"""
        ...


class FlexMatchTrainer(Trainer):
    """Trainer Class for FlexMatch Algorithm"""

    def __init__(
        self,
        name: str,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        val_loader: torch.utils.data.DataLoader,
        train_length: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager: Optional[CheckpointManager]=None,
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name)
        self.trainer_id = "_".join([name, str(uuid.uuid4())])
        self.conf = conf
        self.model = model
        self.train_loaders = train_loaders
        self.train_length = train_length
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()
        self.checkpoint_manager = checkpoint_manager
        self.train_loss_meter = MeanMetric().to(self.conf.device) # Total loss meter
        self.l_train_loss_meter = MeanMetric().to(self.conf.device) # Labeled loss meter
        self.u_train_loss_meter = MeanMetric().to(self.conf.device) # Unlabeled loss meter
        self.val_loss_meter = MeanMetric().to(self.conf.device) # Validation loss meter
        self.f_loss_meter = MeanMetric().to(self.conf.device) # Fraction of confident pseudolabels meter
        self.transforms = get_strong_transforms(resize=self.conf.images.resize)

        # load in target mapping
        if self.conf.metadata.target_mapping_path:
            with open(self.conf.metadata.target_mapping_path, 'r') as f:
                self.map_dict = json.load(f)
            map_arr = np.zeros((len(self.map_dict), 3)).astype(np.uint8)
            for k, v in self.map_dict.items():
                idx = v['class_idx']
                map_arr[idx] = v['rgb'][::-1]

            self.map_arr = map_arr

        if conf.is_main:
            self.run = wandb.init(
                project=conf.wandb.project,
                entity=conf.wandb.entity,
                name=conf.model_run,
                config=OmegaConf.to_container(conf, resolve=True),
            )
        else:
            self.run = None

        # Set up metrics class
        self.train_metrics = MetricLogger(
            name='Train Metrics',
            num_classes=self.conf.model.config.classes,
            device=self.conf.device
        )
        self.val_metrics = MetricLogger(
            name='Validation Metrics',
            num_classes=self.conf.model.config.classes,
            device=self.conf.device
        )

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Train on one batch of labeled and unlabeled images.

        parameters:
        -----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                A tuple containing a batch of labeled images and targets, and a batch of unlabeled images.

        returns:
        --------
            total_loss : torch.Tensor
                The total loss for the batch (labeled + scaled unlabeled).
            l_loss : torch.Tensor
                The labeled loss for the batch.
            scaled_u_loss : torch.Tensor
                The scaled unlabeled loss for the batch.
            f : float
                The fraction of confident pseudo-labels in the unlabeled batch.
        """
        # Unpack batches
        l_batch, u_batch = batch
        l_img, l_targets, _ = l_batch
        weak_img, _ = u_batch

        # Put labeled image and targets on device
        l_img = l_img.to(self.conf.device)
        l_targets = l_targets.to(self.conf.device)

        # Send weak inputs to device and get logits
        weak_inputs = weak_img.float().to(self.conf.device)
        with torch.no_grad():
            weak_logits = self.model(weak_inputs)

        # Pseudo-label the unlabled images (calculated in @torch.no_grad() context)
        beta_c = class_beta(
            weak_logits, 
            tau=self.conf.flexmatch.tau,
            mapping=self.conf.flexmatch.mapping,
            warmup=self.conf.flexmatch.warmup
        )
        tau_vec = beta_c * self.conf.flexmatch.tau
        # tau_vec = torch.tensor(self.conf.flexmatch.tau).repeat(12).to(self.conf.device)  # Temporary fix for tau vector issue

        weak_targets, weak_mask = get_pseudo_labels(tau_vec, weak_logits)

        # Apply strong transforms to weak_img, pseudolabels, and conf_mask
        weak_img = np.moveaxis(weak_img.cpu().numpy(), source=1, destination=3)
        weak_targets = weak_targets.cpu().numpy().astype(np.uint8)
        weak_mask = weak_mask.cpu().numpy().astype(np.uint8)
     
        # Loop through weak transformations, apply strong transforms and output
        strong_img = []
        strong_targets = []
        strong_mask = []
        for img, target, mask in zip(weak_img, weak_targets, weak_mask):
            transformed = self.transforms(image=img, target=target, conf_mask=mask)
            strong_img.append(transformed["image"])
            strong_targets.append(transformed["target"])
            strong_mask.append(transformed["conf_mask"])

        strong_img = torch.stack(strong_img).to(self.conf.device)
        strong_targets = torch.stack(strong_targets).to(self.conf.device)
        strong_mask = torch.stack(strong_mask).bool().to(self.conf.device)

        # Send strong data to device
        inputs = torch.cat((l_img, strong_img)).float().to(self.conf.device)
        l_targets = l_targets.long().to(self.conf.device)
        strong_targets = strong_targets.long().to(self.conf.device)

        # Compute logits for labeled and strong unlabeled images
        concat_logits = self.model(inputs)
        l_logits = concat_logits[: len(l_img)]
        strong_logits = concat_logits[len(l_img) :]

        # Calculate labeled loss
        l_loss = self.criterion(l_logits, l_targets)

        # Calculate the fraction of confident predictions
        f = strong_mask.float().mean().item()

        # Calculate scaled unlabeled loss
        if f > 0:
            u_loss = self.criterion(strong_logits, strong_targets, strong_mask)
            scaled_u_loss = self.conf.flexmatch.lam * f * u_loss
        else:
            scaled_u_loss = torch.tensor(0.0, device=self.conf.device)
            rank_log(self.conf.is_main, self.logger.warning, "No confident pseudo-labels were found. Unlabeled loss contribution is zero.")

        total_loss = l_loss + scaled_u_loss

        # Get the class predictions
        preds = torch.argmax(l_logits, dim=1).to(self.conf.device)

        # Update metrics
        self.train_metrics.update(preds=preds, targets=l_targets)

        return total_loss, l_loss, scaled_u_loss, f

    def _train_epoch(self, epoch: int):
        """ Train over one epoch """
        # Put model in training mode and reset meters
        self.model.train()
        self.train_loss_meter.reset()
        self.u_train_loss_meter.reset()
        self.l_train_loss_meter.reset()
        self.f_loss_meter.reset()
        self.train_metrics.reset()
        
        # Reinstantiate iterator loaders
        train_l_loader, train_u_loader = self.train_loaders
        train_l_loader = iter(train_l_loader)
        train_u_loader = iter(train_u_loader)

        p_bar = tqdm(
            range(self.train_length),
            total=self.train_length,
            colour='yellow',
            disable=not is_main_process()
        )

        # Reinstantiate iterator loaders
        train_l_loader = iter(train_l_loader)
        train_u_loader = iter(train_u_loader)

        for batch_idx in p_bar:

            # Zero the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            # Grab labeled and unlabeled batches
            batch = (next(train_l_loader), next(train_u_loader))

            # Train one batch and backpropagate the errors
            loss, l_loss, u_loss, f = self._train_step(batch)
            loss.backward()

            # Add training losses to MeanMetrics (for unified validation loss over all ranks in DDP)
            l_size = batch[0][0].size(0)
            u_size = batch[1][0].size(0)
            total_size = l_size + u_size

            self.train_loss_meter.update(loss.detach(), weight=total_size)
            self.l_train_loss_meter.update(l_loss.detach(), weight=l_size)
            self.u_train_loss_meter.update(u_loss.detach(), weight=u_size)
            self.f_loss_meter.update(torch.tensor(f, device=self.conf.device), weight=u_size)

            # Step optimizer and update parameters for EMA
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update_params()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Comb. Loss: {loss:.6f}. Conf: {f:.6f}".format(
                    epoch=epoch,
                    epochs=self.conf.training.epochs,
                    batch=batch_idx + 1,
                    iter=self.train_length,
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                    f=f,
                )
            )
            # p_bar.update()

        # ddp barrier
        dist.barrier()

        # Compute avg losses (auto syncs across ranks)
        avg_loss = self.train_loss_meter.compute().item()
        avg_l_loss = self.l_train_loss_meter.compute().item()
        avg_u_loss = self.u_train_loss_meter.compute().item()
        avg_f = self.f_loss_meter.compute().item()

        # Compute epoch metrics and loss
        self.train_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.train_metrics)

        return avg_loss, avg_l_loss, avg_u_loss, avg_f

    @torch.no_grad()
    def _val_step(self, batch: Tuple):
        """ Validate over one batch """

        # Unpack batch and send to device
        img, targets, _ = batch
        inputs = img.float().to(self.conf.device, non_blocking=True)
        targets = targets.long().to(self.conf.device, non_blocking=True)

        # Forward pass through model
        logits = self.model(inputs)

        # Calculate validation loss
        loss = self.criterion(logits, targets)

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.conf.device)

        # Update metrics
        self.val_metrics.update(preds=preds, targets=targets)

        return loss, logits

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        """ Validate over one epoch """

        # Reset meters
        self.model.eval()
        self.val_loss_meter.reset()

        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='blue', disable=not is_main_process())
            
            # Iterate through the batches
            with torch.inference_mode():
                for batch_idx, batch in p_bar:

                    # Validate one batch
                    loss, logits = self._val_step(batch)    

                    # Add validation loss to MeanMetric (for unified validation loss over all ranks in DDP)
                    self.val_loss_meter.update(loss.detach(), weight=logits.size()[0])
                    # Update the progress bar
                    p_bar.set_description(
                        "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                            epoch=epoch + 1,
                            epochs=self.conf.training.epochs,
                            batch=batch_idx + 1,
                            iter=len(self.val_loader),
                            lr=self.scheduler.get_last_lr()[0],
                            loss=loss.item(),
                        )
                    )

        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute epoch metrics
        self.val_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metrics)

        return avg_loss

    def _log_val_images(self, epoch: int) -> list:
        """Run inference on a subset of the val set and return overlay images for wandb."""
        num_images = self.conf.wandb.num_vis_images
        alpha = self.conf.wandb.vis_alpha
        means = np.array(self.conf.metadata.norm.means)
        stds = np.array(self.conf.metadata.norm.std)
        wandb_images = []
        logged = 0
        batches_needed = None

        self.model.eval()
        with apply_ema(self.ema):
            with torch.inference_mode():
                for batch in self.val_loader:
                    img, targets, img_keys = batch
                    batch_size = len(img)
                    if batches_needed is None:
                        batches_needed = math.ceil(num_images / batch_size)

                    inputs = img.float().to(self.conf.device)
                    logits = self.model(inputs)
                    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

                    for i in range(min(batch_size, num_images - logged)):
                        # Denormalize: (3, H, W) -> (H, W, 3) RGB uint8
                        raw = img[i].cpu().numpy()
                        raw = raw * stds[:, None, None] + means[:, None, None]
                        raw = np.clip(raw * 255, 0, 255).astype(np.uint8)
                        raw = np.moveaxis(raw, 0, 2)

                        pred = preds[i]
                        if getattr(self, 'map_arr', None) is not None:
                            colored = self.map_arr[pred][..., ::-1]  # BGR -> RGB
                        else:
                            gray = np.clip(pred * 20, 0, 255).astype(np.uint8)
                            colored = np.stack([gray, gray, gray], axis=-1)

                        overlay = np.clip(alpha * colored + (1 - alpha) * raw, 0, 255).astype(np.uint8)
                        wandb_images.append(wandb.Image(overlay, caption=Path(img_keys[i]).stem))
                        logged += 1

                    if batches_needed is not None and len(wandb_images) >= num_images:
                        break

        return wandb_images

    def train(self):
        """ Train the model using the FlexMatch algorithm """

        rank_log(self.conf.is_main, self.logger.info, f"Training {self.trainer_id} for {self.conf.training.epochs} epochs.")

        for epoch in range(1, self.conf.training.epochs + 1):
            # Train and validate one epoch
            rank_log(self.conf.is_main, self.logger.info, f"TRAINING EPOCH {epoch}")
            train_loss, l_loss, u_loss, avg_f = self._train_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            rank_log(self.conf.is_main, self.logger.info, f"VALIDATING EPOCH {epoch}")
            val_loss = self._val_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            # Logger Logging
            rank_log(
                self.conf.is_main,
                self.logger.info,
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} (Labeled: {l_loss:.6f}, Unlabeled: {u_loss:.6f}) - Val Loss: {val_loss:.6f}"
            )

            avg_metrics = self.val_metrics.results.get('avg', {})

            # Wandb logging
            if self.run is not None:
                val_metrics = self.val_metrics.results
                wandb_log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_labeled_loss": l_loss,
                    "train_unlabeled_loss": u_loss,
                    "pseudo_label_confidence_fraction": avg_f,
                    "val_loss": val_loss,
                    "mc": {key: {} for key in val_metrics['mc'].keys()}
                }

                wandb_log_dict.update({"avg": val_metrics['avg']})

                for class_name, meta in self.map_dict.items():
                    idx = meta['class_idx']
                    for key in wandb_log_dict['mc'].keys():
                        wandb_log_dict['mc'][key][class_name] = val_metrics['mc'][key][idx]

                miou = avg_metrics.get('MeanIoU')
                gds = avg_metrics.get('GeneralizedDiceScore')
                if miou is not None and gds is not None:
                    miou_f = miou.item() if isinstance(miou, torch.Tensor) else float(miou)
                    gds_f = gds.item() if isinstance(gds, torch.Tensor) else float(gds)
                    wandb_log_dict["fitness"] = self.checkpoint_manager.compute_fitness(val_loss, miou_f, gds_f)

                wandb_log_dict["val_predictions"] = self._log_val_images(epoch)
                self.run.log(wandb_log_dict)

            dist.barrier()

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()

            chkpt_logs = {
                "epoch": epoch,
                "val_loss": torch.tensor(val_loss),
                "MeanIoU": avg_metrics.get('MeanIoU'),
                "GeneralizedDiceScore": avg_metrics.get('GeneralizedDiceScore'),
                "model_state_dict": self.model.module.state_dict(),
                "ema_state_dict": ema_state_dict,
            }

            self.checkpoint_manager(logs=chkpt_logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        name: str,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager=Optional[CheckpointManager],
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name) # Initialize the name and AverageMeterSet
        self.trainer_id = "_".join([name, str(uuid.uuid4())])
        self.conf = conf
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()
        self.checkpoint_manager = checkpoint_manager
        self.train_loss_meter = MeanMetric().to(self.conf.device)
        self.val_loss_meter = MeanMetric().to(self.conf.device)

        # Load in target mapping
        if self.conf.metadata.target_mapping_path:
            with open(self.conf.metadata.target_mapping_path, 'r') as f:
                self.map_dict = json.load(f)
            map_arr = np.zeros((len(self.map_dict), 3)).astype(np.uint8)
            for k, v in self.map_dict.items():
                idx = v['class_idx']
                map_arr[idx] = v['rgb'][::-1]

            self.map_arr = map_arr

        if conf.is_main:
            self.run = wandb.init(
                project=conf.wandb.project,
                entity=conf.wandb.entity,
                name=conf.model_run,
                config=OmegaConf.to_container(conf, resolve=True),
            )
        else:
            self.run = None

        # Set up metrics class
        self.train_metric_logger = MetricLogger(
            name='Train Metrics',
            num_classes=self.conf.model.config.classes, 
            device=self.conf.device
        )
        self.val_metric_logger = MetricLogger(
            name='Validation Metrics',
            num_classes=self.conf.model.config.classes, 
            device=self.conf.device
        )

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Train over one batch
        
        parameters:
        -----------
            batch : Tuple[torch.Tensor, torch.Tensor]
                A batch of images and targets from the training DataLoader.
        """
        # Unpack batch and send to device
        img, targets, _ = batch
        inputs = img.to(self.conf.device, non_blocking=True)
        targets = targets.long().to(self.conf.device, non_blocking=True)

        # Forward pass through model
        logits = self.model(inputs)

        # Compute the training loss
        loss = self.criterion(logits, targets)

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.conf.device)

        # Update the training metrics
        self.train_metric_logger.update(preds=preds, targets=targets)

        return loss, logits

    def _train_epoch(self, epoch: int):
        """ Traing over one epoch """
        # Put model in training mode and reset meters
        self.model.train()
        self.train_loss_meter.reset()
        self.train_metric_logger.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            colour='yellow', 
            disable=not is_main_process()
        )

        # Iterate through the batches
        for batch_idx, batch in p_bar:

            # Zero the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            # Train one batch and backpropagate the errors
            loss, logits = self._train_step(batch)
            loss.backward()

            # Add training loss to MeanMetric (for unified validation loss over all ranks in DDP)
            self.train_loss_meter.update(loss.detach(), weight=logits.size()[0])

            # Step optimizer and update parameters for EMA
            self.optimizer.step()

            if self.ema is not None:
                self.ema.update_params()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch,
                    epochs=self.conf.training.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item()
                )
            )

            dist.barrier()
        
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.train_loss_meter.compute().item()

        # Compute epoch metrics and loss
        self.train_metric_logger.compute()
        rank_log(self.conf.is_main, self.logger.info, self.train_metric_logger)

        return avg_loss

    @torch.no_grad()
    def _val_step(self, batch: Tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Validate over one batch """

        # Unpack batch and send to device
        img, targets, img_keys = batch
        inputs = img.float().to(self.conf.device, non_blocking=True)
        targets = targets.long().to(self.conf.device, non_blocking=True)

        # Forward pass through model
        logits = self.model(inputs)

        # Compute validation loss
        loss = self.criterion(logits, targets)

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.conf.device)

        # Update the validation metrics
        self.val_metric_logger.update(preds=preds, targets=targets)

        return loss, logits

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        """ Validate over one epoch """

        # Put model in eval mode and reset meters
        self.model.eval()
        self.val_loss_meter.reset()
        self.val_metric_logger.reset()

        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='blue', disable=not is_main_process())

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:

                    # Validate one batch
                    loss, logits = self._val_step(batch)

                    # Add validation loss to MeanMetric (for unified validation loss over all ranks in DDP)
                    self.val_loss_meter.update(loss.detach(), weight=logits.size()[0])

                    # Update the progress bar
                    p_bar.set_description(
                        "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                            epoch=epoch,
                            epochs=self.conf.training.epochs,
                            batch=batch_idx + 1,
                            iter=len(self.val_loader),
                            lr=self.scheduler.get_last_lr()[0],
                            loss=loss.item(),
                        )
                    )

                    dist.barrier() # DDP barrier to sync after each batch

        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute epoch metrics
        self.val_metric_logger.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metric_logger)

        return avg_loss

    def _log_val_images(self, epoch: int) -> list:
        """Run inference on a subset of the val set and return overlay images for wandb."""
        num_images = self.conf.wandb.num_vis_images
        alpha = self.conf.wandb.vis_alpha
        means = np.array(self.conf.metadata.norm.means)
        stds = np.array(self.conf.metadata.norm.std)
        wandb_images = []
        logged = 0
        batches_needed = None

        self.model.eval()
        with apply_ema(self.ema):
            with torch.inference_mode():
                for batch in self.val_loader:
                    img, targets, img_keys = batch
                    batch_size = len(img)
                    if batches_needed is None:
                        batches_needed = math.ceil(num_images / batch_size)

                    inputs = img.float().to(self.conf.device)
                    logits = self.model(inputs)
                    preds = torch.argmax(logits, dim=1).cpu().numpy().astype(np.uint8)

                    for i in range(min(batch_size, num_images - logged)):
                        # Denormalize: (3, H, W) -> (H, W, 3) RGB uint8
                        raw = img[i].cpu().numpy()
                        raw = raw * stds[:, None, None] + means[:, None, None]
                        raw = np.clip(raw * 255, 0, 255).astype(np.uint8)
                        raw = np.moveaxis(raw, 0, 2)

                        pred = preds[i]
                        if getattr(self, 'map_arr', None) is not None:
                            colored = self.map_arr[pred][..., ::-1]  # BGR -> RGB
                        else:
                            gray = np.clip(pred * 20, 0, 255).astype(np.uint8)
                            colored = np.stack([gray, gray, gray], axis=-1)

                        overlay = np.clip(alpha * colored + (1 - alpha) * raw, 0, 255).astype(np.uint8)
                        wandb_images.append(wandb.Image(overlay, caption=Path(img_keys[i]).stem))
                        logged += 1

                    if batches_needed is not None and len(wandb_images) >= num_images:
                        break

        return wandb_images

    def train(self):
        """ Train the model """
        rank_log(self.conf.is_main, self.logger.info, f"Training {self.trainer_id} for {self.conf.training.epochs} epochs.")

        for epoch in range(1, self.conf.training.epochs + 1):
            # Train and validate one epoch
            rank_log(self.conf.is_main, self.logger.info, f"TRAINING EPOCH {epoch}")
            train_loss = self._train_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            rank_log(self.conf.is_main, self.logger.info, f"VALIDATING EPOCH {epoch}")
            val_loss = self._val_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            # Logger Logging
            rank_log(
                self.conf.is_main,
                self.logger.info,
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            avg_metrics = self.val_metric_logger.results.get('avg', {})

            # Wandb logging
            if self.run is not None:
                val_metrics = self.val_metric_logger.results
                wandb_log_dict = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "mc": {key: {} for key in val_metrics['mc'].keys()}
                }

                wandb_log_dict.update({"avg": val_metrics['avg']})

                for class_name, meta in self.map_dict.items():
                    idx = meta['class_idx']
                    for key in wandb_log_dict['mc'].keys():
                        wandb_log_dict['mc'][key][class_name] = val_metrics['mc'][key][idx]

                miou = avg_metrics.get('MeanIoU')
                gds = avg_metrics.get('GeneralizedDiceScore')
                if miou is not None and gds is not None:
                    miou_f = miou.item() if isinstance(miou, torch.Tensor) else float(miou)
                    gds_f = gds.item() if isinstance(gds, torch.Tensor) else float(gds)
                    wandb_log_dict["fitness"] = self.checkpoint_manager.compute_fitness(val_loss, miou_f, gds_f)

                wandb_log_dict["val_predictions"] = self._log_val_images(epoch)
                self.run.log(wandb_log_dict)

            dist.barrier()

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()
            chkpt_logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(train_loss),
                "val_loss": torch.tensor(val_loss),
                "MeanIoU": avg_metrics.get('MeanIoU'),
                "GeneralizedDiceScore": avg_metrics.get('GeneralizedDiceScore'),
                "model_state_dict": self.model.module.state_dict(),
                "ema_state_dict": ema_state_dict,
            }

            self.checkpoint_manager(logs=chkpt_logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()

        rank_log(self.conf.is_main, self.logger.info, f"Training of {self.trainer_id} completed.")