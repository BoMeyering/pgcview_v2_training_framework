"""
src.trainer.py
Base Trainers Classes
BoMeyering 2025
"""

import torch
import os
import json
import time
import cv2
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
from torch.utils.tensorboard import SummaryWriter
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

    def __init__(self, name: str, tb_writer: SummaryWriter=None):
        super().__init__()
        self.name = name
        self.tb_writer = tb_writer

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
        tb_writer: SummaryWriter,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loaders: Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader],
        val_loader: torch.utils.data.DataLoader,
        train_length: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager: Optional[CheckpointManager]=None,
        sanity_check: bool=True,
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name, tb_writer=tb_writer)
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
        self.sanity_check = sanity_check
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
        # beta_c = class_beta(
        #     weak_logits, 
        #     tau=self.conf.flexmatch.tau,
        #     mapping=self.conf.flexmatch.mapping,
        #     warmup=self.conf.flexmatch.warmup
        # )
        # tau_vec = beta_c * self.conf.flexmatch.tau
        tau_vec = torch.tensor(self.conf.flexmatch.tau).repeat(12).to(self.conf.device)  # Temporary fix for tau vector issue

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

            # Tensorboard batch writing
            batch_step = ((epoch-1) * self.train_length) + batch_idx
            if dist.get_rank() == 0:
                self.tb_writer.add_scalar(
                    tag="batch_loss/train", scalar_value=loss.item(), global_step=batch_step
                )
                self.tb_writer.add_scalar(
                    tag="batch_loss/train_labeled", scalar_value=l_loss.item(), global_step=batch_step
                )
                self.tb_writer.add_scalar(
                    tag="batch_loss/train_unlabeled", scalar_value=u_loss.item(), global_step=batch_step
                )
                self.tb_writer.add_scalar(
                    tag="batch_loss/fraction_confident", scalar_value=f, global_step=batch_step
                )

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

        # Tensorboard epoch logging
        if dist.get_rank() == 0:
            self.tb_writer.add_scalar(
                tag="epoch_loss/train", scalar_value=avg_loss, global_step=epoch
            )
            self.tb_writer.add_scalar(
                tag="epoch_loss/train_labeled", scalar_value=avg_l_loss, global_step=epoch
            )
            self.tb_writer.add_scalar(
                tag="epoch_loss/train_unlabeled", scalar_value=avg_u_loss, global_step=epoch
            )
            self.tb_writer.add_scalar(
                tag="epoch_fraction_confident", scalar_value=avg_f, global_step=epoch
            )
            self._tb_log_metrics(
                self.train_metrics.results, 
                main_tag="train_metrics", 
                global_step=epoch, 
                exclude_idx=self.conf.tb_exclude_classes
            )

        return avg_loss, avg_l_loss, avg_u_loss

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

                    # Tensorboard batch writing
                    batch_step = ((epoch-1) * len(self.val_loader)) + batch_idx
                    if dist.get_rank() == 0:
                        self.tb_writer.add_scalar(
                            tag="batch_loss/val", scalar_value=loss.item(), global_step=batch_step
                        )
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute epoch metrics
        self.val_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metrics)

        # Tensorboard epoch logging
        if dist.get_rank() == 0:
            self.tb_writer.add_scalar(
                tag="epoch_loss/val", scalar_value=avg_loss, global_step=epoch
            )
            self._tb_log_metrics(
                self.val_metrics.results, 
                main_tag="val_metrics", 
                global_step=epoch,
                exclude_idx=self.conf.tb_exclude_classes
            )

        return avg_loss

    def _sanity_check(self, epoch):
        """ Run a sanity check for the model """
        rank_log(self.conf.is_main, self.logger.info, f"SANITY CHECK {epoch}")

        out_dir = Path(self.conf.directories.output_dir) / self.conf.model_run / "_".join(["epoch", str(epoch)])
        if self.conf.local_rank == 0:
            os.makedirs(out_dir)
            
        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='green', disable=not is_main_process())

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:
                    if batch_idx % 10 == 0:
                        # Unpack batch and send to device
                        img, targets, img_keys = batch
                        inputs = img.to(self.conf.device, non_blocking=True)
                        targets = targets.long().to(self.conf.device, non_blocking=True)

                        # Forward pass through model
                        logits = self.model(inputs)

                        maps = torch.argmax(logits, dim=1)

                        # for i, img in enumerate(maps):
                        for img_key, img in zip(img_keys, maps):
                            img = img.detach().cpu().numpy().astype(np.uint8)
                            if getattr(self, 'map_arr', None) is not None:
                                img = self.map_arr[img]
                            else:
                                img *= 20 # Scale outputs to make class distinction clear

                            cv2.imwrite(str(Path(out_dir) / f"{Path(img_key).stem}_outmap.png"), img)
                        
                        # Update the progress bar
                        p_bar.set_description(
                            "Sanity Check: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.".format(
                                epoch=epoch,
                                epochs=self.conf.training.epochs,
                                batch=batch_idx + 1,
                                iter=len(self.val_loader)
                            )
                        )


    def _tb_log_metrics(self, metric_dict: dict, main_tag: str, global_step: int, exclude_idx: Optional[List[int]]=None):
        """ Log metrics from a metric dictionary to TensorBoard """
        for type, v in metric_dict.items(): # type is 'avg' or 'mc'
            if type == 'avg':
                for mk, mv in v.items(): # mk is the metric name, mv is the metric value as a torch.Tensor
                    self.tb_writer.add_scalar(f"{main_tag}/avg_{mk}", mv.item(), global_step=global_step)
            elif type == 'mc':
                metric_map = {data['class_idx']: cname for cname, data in self.map_dict.items()} # Create a mapping from class index (int) to class name (str)
                for mk, mv in v.items():
                    scalar_dict = {metric_map.get(i): t.item() for i, t in enumerate(mv) if i not in exclude_idx} # Map the tensor values to a new dict with class names as keys
                    for sk, sv in scalar_dict.items():
                        self.tb_writer.add_scalar(f"{main_tag}/{sk}_{mk}", sv, global_step=global_step)

    def train(self):
        """ Train the model using the FlexMatch algorithm """

        rank_log(self.conf.is_main, self.logger.info, f"Training {self.trainer_id} for {self.conf.training.epochs} epochs.")

        for epoch in range(1, self.conf.training.epochs + 1):
            # Train and validate one epoch
            rank_log(self.conf.is_main, self.logger.info, f"TRAINING EPOCH {epoch}")
            train_loss, l_loss, u_loss = self._train_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            rank_log(self.conf.is_main, self.logger.info, f"VALIDATING EPOCH {epoch}")
            val_loss = self._val_epoch(epoch)
            time.sleep(1)
            dist.barrier()

            if self.sanity_check:
                self._sanity_check(epoch)
                time.sleep(1)
                dist.barrier()
            
            # Logger Logging
            rank_log(
                self.conf.is_main, 
                self.logger.info,
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} (Labeled: {l_loss:.6f}, Unlabeled: {u_loss:.6f}) - Val Loss: {val_loss:.6f}"
            )

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()

            chkpt_logs = {
                "epoch": epoch,
                "val_loss": torch.tensor(val_loss),
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
        tb_writer: SummaryWriter,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        checkpoint_manager=Optional[CheckpointManager],
        sanity_check: bool=True,
        ema: Optional[EMA]=None,
    ):
        super().__init__(name=name, tb_writer=tb_writer) # Initialize the name and AverageMeterSet
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
        self.sanity_check = sanity_check
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
        self.train_metrics.update(preds=preds, targets=targets)

        return loss, logits

    def _train_epoch(self, epoch: int):
        """ Traing over one epoch """
        # Put model in training mode and reset meters
        self.model.train()
        self.train_loss_meter.reset()

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

            # Tensorboard batch writing
            batch_step = ((epoch-1) * len(self.train_loader)) + batch_idx
            if dist.get_rank() == 0:
                self.tb_writer.add_scalar(
                    tag="batch_loss/train", scalar_value=loss.item(), global_step=batch_step
                )
        
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.train_loss_meter.compute().item()

        # Compute epoch metrics and loss
        self.train_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.train_metrics)
        
        # Tensorboard epoch logging
        if dist.get_rank() == 0:
            self.tb_writer.add_scalar(
                tag="epoch_loss/train", scalar_value=avg_loss, global_step=epoch
            )

            self._tb_log_metrics(
                self.train_metrics.results, 
                main_tag="train_metrics", 
                global_step=epoch, 
                exclude_idx=self.conf.tb_exclude_classes
            )

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
        self.val_metrics.update(preds=preds, targets=targets)

        return loss, logits

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        """ Validate over one epoch """

        # Put model in eval mode and reset meters
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
                            epoch=epoch,
                            epochs=self.conf.training.epochs,
                            batch=batch_idx + 1,
                            iter=len(self.val_loader),
                            lr=self.scheduler.get_last_lr()[0],
                            loss=loss.item(),
                        )
                    )

                    # Tensorboard batch writing
                    batch_step = ((epoch-1) * len(self.val_loader)) + batch_idx
                    if dist.get_rank() == 0:
                        self.tb_writer.add_scalar(
                            tag="batch_loss/val", scalar_value=loss.item(), global_step=batch_step
                        )
        # ddp barrier
        dist.barrier()

        # Compute avg loss (auto syncs across ranks)
        avg_loss = self.val_loss_meter.compute().item()

        # Compute epoch metrics
        self.val_metrics.compute()
        rank_log(self.conf.is_main, self.logger.info, self.val_metrics)

        # Tensorboard epoch logging
        if dist.get_rank() == 0:
            self.tb_writer.add_scalar(
                tag="epoch_loss/val", scalar_value=avg_loss, global_step=epoch
            )
            self._tb_log_metrics(
                self.val_metrics.results, 
                main_tag="val_metrics", 
                global_step=epoch,
                exclude_idx=self.conf.tb_exclude_classes
            )

        return avg_loss

    def _sanity_check(self, epoch):
        """ Run a sanity check for the model """
        rank_log(self.conf.is_main, self.logger.info, f"SANITY CHECK {epoch}")

        out_dir = Path(self.conf.directories.output_dir) / self.conf.model_run / "_".join(["epoch", str(epoch)])
        if self.conf.local_rank == 0:
            os.makedirs(out_dir)
            
        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='green', disable=not is_main_process())

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:
                    if batch_idx % 10 == 0:
                        # Unpack batch and send to device
                        img, targets, img_keys = batch
                        inputs = img.to(self.conf.device, non_blocking=True)
                        targets = targets.long().to(self.conf.device, non_blocking=True)

                        # Forward pass through model
                        logits = self.model(inputs)

                        maps = torch.argmax(logits, dim=1)

                        # for i, img in enumerate(maps):
                        for img_key, img in zip(img_keys, maps):
                            img = img.detach().cpu().numpy().astype(np.uint8)
                            if getattr(self, 'map_arr', None) is not None:
                                img = self.map_arr[img]
                            else:
                                img *= 20 # Scale outputs to make class distinction clear

                            cv2.imwrite(str(Path(out_dir) / f"{Path(img_key).stem}_outmap.png"), img)
                        
                        # Update the progress bar
                        p_bar.set_description(
                            "Sanity Check: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}.".format(
                                epoch=epoch,
                                epochs=self.conf.training.epochs,
                                batch=batch_idx + 1,
                                iter=len(self.val_loader)
                            )
                        )

    def _tb_log_metrics(self, metric_dict: dict, main_tag: str, global_step: int, exclude_idx: Optional[List[int]]=None):
        """ Log metrics from a metric dictionary to TensorBoard """
        for type, v in metric_dict.items(): # type is 'avg' or 'mc'
            if type == 'avg':
                for mk, mv in v.items(): # mk is the metric name, mv is the metric value as a torch.Tensor
                    self.tb_writer.add_scalar(f"{main_tag}/avg_{mk}", mv.item(), global_step=global_step)
            elif type == 'mc':
                metric_map = {data['class_idx']: cname for cname, data in self.map_dict.items()} # Create a mapping from class index (int) to class name (str)
                for mk, mv in v.items():
                    scalar_dict = {metric_map.get(i): t.item() for i, t in enumerate(mv) if i not in exclude_idx} # Map the tensor values to a new dict with class names as keys
                    for sk, sv in scalar_dict.items():
                        self.tb_writer.add_scalar(f"{main_tag}/{sk}_{mk}", sv, global_step=global_step)

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

            if self.sanity_check:
                self._sanity_check(epoch)
                time.sleep(1)
                dist.barrier()

            # Logger Logging
            rank_log(
                self.conf.is_main,
                self.logger.info, 
                f"Epoch {epoch} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
            )

            with apply_ema(self.ema):
                # Create checkpoint logs
                ema_state_dict = self.model.module.state_dict()

            chkpt_logs = {
                "epoch": epoch,
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.module.state_dict(),
                "ema_state_dict": ema_state_dict,
            }

            self.checkpoint_manager(logs=chkpt_logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()

