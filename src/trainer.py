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
from src.flexmatch import get_pseudo_labels
from src.parameters import EMA, apply_ema
# from src.callbacks import ModelCheckpoint
from src.metrics import MetricLogger, MeterSet, RunningAvgMeter, ValueMeter
from src.transforms import get_strong_transforms
from src.distributed import is_main_process
from src.utils.loggers import rank_log
from src.callbacks import CheckpointManager

class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self, name: str, meter_set: MeterSet, tb_writer: SummaryWriter=None):
        super().__init__()
        self.name = name
        self.meters = meter_set
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
        name,
        args: argparse.Namespace,
        model: torch.nn.Module,
        train_loaders,
        train_length,
        val_loader,
        optimizer,
        criterion,
        train_samplers=None,
        scheduler=None,
        ema=None,
        tb_logger: SummaryWriter = None,
        class_map: dict = None,
    ):
        super().__init__(name=name)
        self.trainer_id = "_".join(["fixmatch", str(uuid.uuid4())])
        self.args = args
        self.model = model
        self.train_loaders = train_loaders
        self.train_length = train_length
        self.train_samplers = train_samplers
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        self.logger = logging.getLogger()
        self.tb_logger = tb_logger
        self.class_map = class_map
        self.transforms = get_strong_transforms(resize=args.model.resize)

        # setup metrics class
        self.train_metrics = MetricLogger(
            num_classes=args.model.num_classes, device=args.device
        )
        self.val_metrics = MetricLogger(
            num_classes=args.model.num_classes, device=args.device
        )

        # chkpt_path = Path(self.args.directories.chkpt_dir) / self.args.run_name
        # self.checkpoint = ModelCheckpoint(filepath=chkpt_path, metadata=vars(self.args))

    def _train_step(self, batch: Tuple):
        "Train on one batch of labeled and unlabeled images."
        # Unpack batches
        l_batch, u_batch = batch
        l_img, l_targets = l_batch
        weak_img = u_batch

        # Put labeled image and targets on device
        l_img = l_img.to(self.args.device)
        l_targets = l_targets.to(self.args.device)

        # Send weak inputs to device and get logits
        weak_inputs = weak_img.float().to(self.args.device)
        with torch.no_grad():
            weak_logits = self.model(weak_inputs)

        # Pseudo-label the unlabled images (calculated in @torch.no_grad() context)
        weak_targets, weak_mask = get_pseudo_labels(self.args, weak_logits)

        # Apply strong transforms to weak_img, pseudolabels, and conf_mask
        weak_img_np = np.moveaxis(weak_img.cpu().numpy(), source=1, destination=3)
        weak_targets = weak_targets.cpu().numpy().astype(np.uint8)
        weak_mask = weak_mask.cpu().numpy().astype(np.uint8)

        # Loop through weak transformations, apply strong transforms and output
        strong_img = []
        strong_targets = []
        strong_mask = []
        for zipped in zip(weak_img_np, weak_targets, weak_mask):
            img, target, mask = zipped
            transformed = self.transforms(image=img, target=target, conf_mask=mask)
            strong_img.append(transformed["image"])
            strong_targets.append(transformed["target"])
            strong_mask.append(transformed["conf_mask"])

        strong_img = torch.stack(strong_img).to(self.args.device)
        strong_targets = torch.stack(strong_targets).to(self.args.device)
        strong_mask = torch.stack(strong_mask).bool().to(self.args.device)

        # Send strong data to device
        inputs = torch.cat((l_img, strong_img)).float().to(self.args.device)
        l_targets = l_targets.long().to(self.args.device)
        strong_targets = strong_targets.long().to(self.args.device)

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
            scaled_u_loss = self.args.fixmatch.lam * f * u_loss
        else:
            scaled_u_loss = torch.tensor(0.0, device=self.args.device)
            if self.rank == 0:
                self.logger.warning(
                    "No confident pseudo-labels were found. Unlabeled loss contribution is zero."
                )

        total_loss = l_loss + scaled_u_loss

        # Update loss meters
        self.meters.update("total_loss", total_loss.item(), 1)
        self.meters.update("labeled_loss", l_loss.item(), l_logits.size()[0])
        self.meters.update(
            "unlabeled_loss", scaled_u_loss.item(), strong_logits.size()[0]
        )

        # Get the class predictions
        preds = torch.argmax(l_logits, dim=1).to(self.args.device)

        # Update metrics
        self.train_metrics.update(preds=preds, targets=l_targets)

        return total_loss, l_loss, scaled_u_loss, f

    def _train_epoch(self, epoch: int):

        # Reset meters
        self.model.train()
        self.meters.reset()
        self.train_metrics.reset()
        self.val_metrics.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(self.train_length), colour='yellow')
        train_l_loader, train_u_loader = self.train_loaders

        # Reinstantiate iterator loaders
        train_l_loader = iter(train_l_loader)
        train_u_loader = iter(train_u_loader)

        for batch_idx in range(self.train_length):

            # Zero the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            # Get batches
            batches = (next(train_l_loader), next(train_u_loader))

            # Train one batch and backpropagate
            loss, l_loss, u_loss, f = self._train_step(batches)
            loss.backward()

            # Step optimizer and update parameters for EMA
            self.optimizer.step()
            if self.ema:
                self.ema.update()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Comb. Loss: {loss:.6f}. Conf: {f:.6f}".format(
                    epoch=epoch,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=self.train_length,
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                    f=f,
                )
            )
            p_bar.update()

            # Tensorboard batch writing
            loss_dict = {
                "train_loss": loss,
                "train_labeled_loss": l_loss,
                "train_unlabeled_loss": u_loss,
            }
            f_dict = {"p_confident": f}
            batch_step = (epoch * self.train_length) + batch_idx
            if self.rank == 0:
                self.tb_logger.log_scalar_dict(
                    main_tag="step_loss", scalar_dict=loss_dict, step=batch_step
                )
                self.tb_logger.log_scalar_dict(
                    main_tag="step_p", scalar_dict=f_dict, step=batch_step
                )

            if batch_idx % 200 == 0:
                avg_metrics, mc_metrics = self.train_metrics.compute()
                print(avg_metrics)
                print(mc_metrics)

        # Step LR scheduler
        if self.scheduler:
            self.scheduler.step()

        # Compute epoch metrics and loss
        avg_metrics, mc_metrics = self.train_metrics.compute()
        loss = self.meters["total_loss"].avg
        l_loss = self.meters["labeled_loss"].avg
        u_loss = self.meters["unlabeled_loss"].avg

        # Epoch Loss Logging
        loss_dict = {
            "train_loss": loss,
            "train_labeled_loss": l_loss,
            "train_unlabeled_loss": u_loss,
        }

        if self.rank == 0:
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
            )

            # Epoch Average Metric Logging
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_train_metrics", scalar_dict=avg_metrics, step=epoch_step
            )

            # Epoch Multiclass Metric Logging
            self.tb_logger.log_tensor_dict(
                main_tag="epoch_train_metrics",
                tensor_dict=mc_metrics,
                step=epoch_step,
                class_map=self.class_map,
            )

            # Logger Logging
            self.logger.info(
                f"Epoch {epoch + 1} - Total Loss: {loss:.6f} Labeled Loss: {l_loss:.6f} Unlabeled Loss: {u_loss:.6f}"
            )
            self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
            self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        return loss, l_loss, u_loss

    @torch.no_grad()
    def _val_step(self, batch: Tuple):

        # Unpack batch and send to device
        img, targets = batch
        img = img.float().to(self.args.device)
        targets = targets.long().to(self.args.device)

        # Forward pass through model
        logits = self.model(img)

        # Calculate validation loss
        loss = self.criterion(logits, targets)

        # Update running meters
        self.meters.update("validation_loss", loss.item(), logits.size()[0])

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.args.device)

        # Update metrics
        self.val_metrics.update(preds=preds, targets=targets)

        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int):

        # Reset meters
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)), colour='blue')
        for batch_idx, batch in enumerate(self.val_loader):

            # Validate one batch
            loss = self._val_step(batch)

            # Update the progress bar
            p_bar.set_description(
                "Val Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.args.model.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.val_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                )
            )
            p_bar.update()

        # Compute epoch metrics
        avg_metrics, mc_metrics = self.val_metrics.compute()
        loss = self.meters["validation_loss"].avg

        # Epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = {"validation_loss": loss}

        if self.rank == 0:
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
            )

            # Epoch Average Validation Metric Logging
            self.tb_logger.log_scalar_dict(
                main_tag="epoch_val_metrics", scalar_dict=avg_metrics, step=epoch_step
            )

            # Epoch Multiclass Validation Metric Logging
            self.tb_logger.log_tensor_dict(
                main_tag="epoch_val_metrics",
                tensor_dict=mc_metrics,
                step=epoch_step,
                class_map=self.class_map,
            )

            # Logger Logging
            self.logger.info(f"Epoch {epoch + 1} - Validation Loss: {loss:.6f}")
            self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
            self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        return loss

    def train(self):
        if self.rank == 0:
            self.logger.info(
                f"Training {self.trainer_id} for {self.args.model.epochs} epochs."
            )
        for epoch in range(self.args.model.epochs):

            # Update the epoch in the DistributedSamplers
            if self.train_samplers is not None:
                for sampler in self.train_samplers:
                    sampler.set_epoch(epoch)
            # Train and validate one epoch
            train_loss = self._train_epoch(epoch)
            val_loss = self._val_epoch(epoch)

            logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(train_loss[0]),
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.state_dict(),
            }

            # self.logger.info(f"Epoch {logs['epoch'] + 1} Avg Loss, Train Loss: {logs['train_loss'].item()}, Val Loss: {logs['val_loss'].item()}")
            
            # self.checkpoint(epoch=epoch, logs=logs)

class SupervisedTrainer(Trainer):
    def __init__(
        self,
        name: str,
        meter_set: MeterSet,
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
        super().__init__(name=name, meter_set=meter_set, tb_writer=tb_writer) # Initialize the name and AverageMeterSet
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
        self.train_loss_metric = MeanMetric().to(self.conf.device)
        self.val_loss_metric = MeanMetric().to(self.conf.device)

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
        self.meters.reset()
        self.train_loss_metric.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), colour='yellow', disable=not is_main_process())

        # Iterate through the batches
        for batch_idx, batch in p_bar:

            # Zero the optimizer
            self.optimizer.zero_grad(set_to_none=True)

            # Train one batch and backpropagate the errors
            loss, logits = self._train_step(batch)
            loss.backward()

            # Update the training loss meter after loss.backward() so all loss has been all-reduced
            update_dict = {
                'train_loss_smooth': {'val': loss.item(), 'n': 1}
            }
            self.meters.update(update_dict)

            # Add training loss to MeanMetric (for unified validation loss over all ranks in DDP)
            self.train_loss_metric.update(loss.detach(), weight=logits.size()[0])

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
                    loss=self.meters['train_loss_smooth'].mean
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
        avg_loss = self.train_loss_metric.compute().item()

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
        inputs = img.to(self.conf.device, non_blocking=True)
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
        self.meters.reset()
        self.val_loss_metric.reset()

        with apply_ema(self.ema):
            # Set progress bar and unpack batches
            p_bar = tqdm(enumerate(self.val_loader), total=len(self.val_loader), colour='blue', disable=not is_main_process())

            # Iterate through the batches
            with torch.inference_mode():  
                for batch_idx, batch in p_bar:

                    # Validate one batch
                    loss, logits = self._val_step(batch)

                    # Update the validation loss meter
                    update_dict = {
                        'val_loss_smooth': {'val': loss.item(), 'n': 1}
                    }
                    self.meters.update(update_dict)

                    # Add validation loss to MeanMetric (for unified validation loss over all ranks in DDP)
                    self.val_loss_metric.update(loss.detach(), weight=logits.size()[0])

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
        avg_loss = self.val_loss_metric.compute().item()

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

            # Logger Logging
            time.sleep(1)
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            dist.barrier()

            logs = {
                "epoch": epoch,
                "train_loss": torch.tensor(train_loss),
                "val_loss": torch.tensor(val_loss),
                "model_state_dict": self.model.state_dict(),
            }

            self.checkpoint_manager(logs=logs)

            # Step LR scheduler
            if self.scheduler:
                self.scheduler.step()

