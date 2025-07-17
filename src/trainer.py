"""
src.trainer.py
Base Trainers Classes
BoMeyering 2025
"""

import torch

from glob import glob
import uuid
import logging
import argparse
from typing import Tuple
from pathlib import Path
from abc import ABC, abstractmethod
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
import torch.distributed as dist
import numpy as np

from typing import Union
from torch.utils.tensorboard import SummaryWriter
from src.eval import AverageMeterSet
from src.flexmatch import get_pseudo_labels
# from src.callbacks import ModelCheckpoint
from src.metrics import MetricLogger
from src.transforms import get_strong_transforms


class Trainer(ABC):
    """Abstract Trainer Class"""

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.meters = AverageMeterSet()

    @abstractmethod
    def _train_step(self, batch):
        """Implement the train step for one batch"""

    @abstractmethod
    def _val_step(self, batch):
        """Implement the val step for one batch"""

    @abstractmethod
    def _train_epoch(self, epoch):
        """Implement the training method for one epoch"""

    @abstractmethod
    def _val_epoch(self, epoch):
        """Implement the validation method for one epoch"""

    @abstractmethod
    def train(self):
        """Implement the whole training loop"""


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

        # try:
        #     self.rank = dist.get_rank()
        # except ValueError:
        #     self.rank = 0
        # except RuntimeError:
        #     self.rank = 0

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
        p_bar = tqdm(range(self.train_length))
        train_l_loader, train_u_loader = self.train_loaders

        # Reinstantiate iterator loaders
        train_l_loader = iter(train_l_loader)
        train_u_loader = iter(train_u_loader)

        for batch_idx in range(self.train_length):

            # Zero the optimizer
            self.optimizer.zero_grad()

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
                    epoch=epoch + 1,
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

        # Set the epoch step
        epoch_step = epoch + 1

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
        p_bar = tqdm(range(len(self.val_loader)))
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
        name,
        conf: OmegaConf,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        ema=None,
        # tb_logger: torch.utils.tensorboard.writer.SummaryWriter = None,
        class_map: dict = None,
    ):
        super().__init__(name=name) # Initialize the name and AverageMeterSet
        self.trainer_id = "_".join(["supervised_trainer", str(uuid.uuid4())])
        self.conf = conf
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.ema = ema
        # self.logger = logging.getLogger()
        # self.tb_logger = tb_logger
        self.class_map = class_map

        # torch.distributed stuff
        # try:
        #     self.rank = dist.get_rank()
        # except ValueError:
        #     self.rank = 0
        # except RuntimeError:
        #     self.rank = 0

        # Set up metrics class
        self.train_metrics = MetricLogger(
            num_classes=self.conf.model.config.classes, device=self.conf.device
        )
        self.val_metrics = MetricLogger(
            num_classes=self.conf.model.config.classes, device=self.conf.device
        )

        checkpoint_path = Path(self.conf.directories.checkpoint_dir) / self.conf.model_run
        # self.checkpoint = ModelCheckpoint(filepath=chkpt_path, metadata=vars(self.conf), monitor='train_loss')

    def _train_step(self, batch: Tuple):
        """ Train over one batch """
        # Unpack batch and send to device
        img, targets = batch
        inputs = img.to(self.conf.device)
        targets = targets.long().to(self.conf.device)

        # Forward pass through model
        logits = self.model(inputs)

        # Compute the training loss
        loss = self.criterion(logits, targets)

        # Update the training loss meter
        self.meters.update("train_loss", loss.item())

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.conf.device)

        # Update the training metrics
        self.train_metrics.update(preds=preds, targets=targets)

        return loss

    def _train_epoch(self, epoch: int):
        """ Traing over one epoch """
        # Put model in training mode and reset meters
        self.model.train()
        self.meters.reset()
        self.train_metrics.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.train_loader)))

        # Iterate through the batches
        for batch_idx, batch in enumerate(self.train_loader):

            # Zero the optimizer
            self.optimizer.zero_grad()

            # Train one batch and backpropagate the errors
            loss = self._train_step(batch)
            loss.backward()

            # Step optimizer and update parameters for EMA
            self.optimizer.step()
            if self.ema:
                self.ema.update()

            # Update progress bar
            p_bar.set_description(
                "Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.6f}. Loss: {loss:.6f}".format(
                    epoch=epoch + 1,
                    epochs=self.conf.training.epochs,
                    batch=batch_idx + 1,
                    iter=len(self.train_loader),
                    lr=self.scheduler.get_last_lr()[0],
                    loss=loss.item(),
                )
            )
            p_bar.update()

            # Tensorboard batch writing
            # loss_dict = {"train_loss": loss}
            # batch_step = (epoch * len(train_loader)) + batch_idx
            # if self.rank == 0:
            #     self.tb_logger.log_scalar_dict(
            #         main_tag="step_loss", scalar_dict=loss_dict, step=batch_step
            #     )

            # if batch_idx % 200 == 0:
            #     avg_metrics, mc_metrics = self.train_metrics.compute()
            #     print(avg_metrics)
            #     print(mc_metrics)

        # Step LR scheduler
        if self.scheduler:
            self.scheduler.step()

        # Compute epoch metrics and loss
        # avg_metrics, mc_metrics = self.train_metrics.compute()
        # avg_loss = self.meters["train_loss"].avg

        # Set the epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging if not in distributed training
        # loss_dict = tag_scalar_dict = {"train_loss": avg_loss}
        # if self.rank == 0:
        #     self.tb_logger.log_scalar_dict(
        #         main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
        #     )

        #     # Epoch Average Metric Logging
        #     self.tb_logger.log_scalar_dict(
        #         main_tag="epoch_train_metrics", scalar_dict=avg_metrics, step=epoch_step
        #     )

        #     # Epoch Multiclass Metric Logging
        #     self.tb_logger.log_tensor_dict(
        #         main_tag="epoch_train_metrics",
        #         tensor_dict=mc_metrics,
        #         step=epoch_step,
        #         class_map=self.class_map,
        #     )

        #     # Logger Logging
        #     self.logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.6f}")
        #     self.logger.info(f"Epoch {epoch + 1} - Avg Metrics {avg_metrics}")
        #     self.logger.info(f"Epoch {epoch + 1} - Multiclass Metrics {mc_metrics}")

        # return avg_loss
        return True

    @torch.no_grad()
    def _val_step(self, batch: Tuple):
        """ Validate over one batch """

        # Unpack batch and send to device
        img, targets = batch
        inputs = img.to(self.conf.device)
        targets = targets.long().to(self.conf.device)

        # Forward pass through model
        logits = self.model(inputs)

        # Compute validation loss
        loss = self.criterion(logits, targets)

        # Update the validation loss meter
        self.meters.update("validation_loss", loss.item(), logits.size()[0])

        # Get the class predictions
        preds = torch.argmax(logits, dim=1).to(self.conf.device)

        # Update the validation metrics
        self.val_metrics.update(preds=preds, targets=targets)

        return loss

    @torch.no_grad()
    def _val_epoch(self, epoch: int):
        """ Validate over one epoch """
        # Put model in eval mode and reset meters
        self.model.eval()
        self.meters.reset()

        # Set progress bar and unpack batches
        p_bar = tqdm(range(len(self.val_loader)))

        # Iterate through the batches
        for batch_idx, batch in enumerate(self.val_loader):

            # Validate one batch
            loss = self._val_step(batch)

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
            p_bar.update()

        # Compute epoch metrics
        avg_metrics, mc_metrics = self.val_metrics.compute()
        avg_loss = self.meters["validation_loss"].avg

        # Epoch step
        epoch_step = epoch + 1

        # Epoch Loss Logging
        loss_dict = {"validation_loss": avg_loss}

        # if self.rank == 0:
        #     self.tb_logger.log_scalar_dict(
        #         main_tag="epoch_loss", scalar_dict=loss_dict, step=epoch_step
        #     )

        #     # Epoch Average Validation Metric Logging
        #     self.tb_logger.log_scalar_dict(
        #         main_tag="epoch_val_metrics", scalar_dict=avg_metrics, step=epoch_step
        #     )

        #     # Epoch Multiclass Validation Metric Logging
        #     self.tb_logger.log_tensor_dict(
        #         main_tag="epoch_val_metrics",
        #         tensor_dict=mc_metrics,
        #         step=epoch_step,
        #         class_map=self.class_map
        #     )

        #     # Update the epoch in the DistributedSampler
        #     if self.train_sampler is not None:
        #         self.train_sampler.set_epoch(epoch)


        #     # Train and validate one epoch
        #     train_loss = self._train_epoch(epoch)
        #     # val_loss = self._val_epoch(epoch)

        #     logs = {
        #         "epoch": epoch,
        #         "train_loss": torch.tensor(train_loss),
        #         # "val_loss": torch.tensor(val_loss),
        #         "model_state_dict": self.model.state_dict(),
        #     }

            

        #     self.checkpoint(epoch=epoch, logs=logs)

    def train(self):
        """ Train the model """
        # if self.rank == 0:
        #     self.logger.info(
        #         f"Training {self.trainer_id} for {self.conf.model.epochs} epochs."
        #     )
        for epoch in range(self.conf.training.epochs):
            # Train and validate one epoch
            # self.logger.info(f"TRAINING EPOCH {epoch + 1}")
            train_loss = self._train_epoch(epoch)
            # self.logger.info(f"VALIDATING EPOCH {epoch + 1}")
            val_loss = self._val_epoch(epoch)

            # logs = {
            #     "epoch": epoch,
            #     "train_loss": torch.tensor(train_loss),
            #     "val_loss": torch.tensor(val_loss),
            #     "model_state_dict": self.model.state_dict(),
            # }

            # self.logger.info(f"Epoch {logs['epoch'] + 1} Avg Loss, Train Loss: {logs['train_loss'].item()}, Val Loss: {logs['val_loss'].item()}")

            # self.checkpoint(epoch=epoch, logs=logs)

class TBLogger:
    def __init__(self, writer: torch.utils.tensorboard.writer.SummaryWriter):
        self.writer = writer

    def log_scalar_dict(self, main_tag, scalar_dict, step):
        # Epoch Average Metric Logging
        for k, v in scalar_dict.items():
            self.writer.add_scalar(
                tag=f"{main_tag}/{k}", scalar_value=v, global_step=step, new_style=True
            )

    def log_tensor_dict(self, main_tag, tensor_dict, step, class_map):
        # Log each tensor value for each key in the tensor_dict
        for k, v in tensor_dict.items():
            for i, tensor_value in enumerate(v):
                class_name = class_map[str(i)]
                self.writer.add_scalar(
                    tag=f"{main_tag}/{k}/{class_name}",
                    scalar_value=tensor_value,
                    global_step=step,
                    new_style=True,
                )