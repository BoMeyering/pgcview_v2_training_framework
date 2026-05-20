"""
src.callbacks.py
Callback Functions
BoMeyering 2025
"""

import torch
import os
import logging
from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime
from src.utils.loggers import rank_log

class CheckpointManager:
    def __init__(self, conf: OmegaConf, top_k: int=5, w_loss: float=0.5, w_miou: float=1.0, w_gds: float=1.0):
        self.checkpoint_dir = Path(conf.directories.checkpoint_dir)
        self.model_run_name = conf.model_run
        self.logger = logging.getLogger()
        self.conf = conf
        self.top_k = top_k
        self.w_loss = w_loss
        self.w_miou = w_miou
        self.w_gds = w_gds
        self.top_checkpoints = []  # list of (fitness, filepath), sorted descending
        self.most_recent_checkpoint = []

        self.checkpoint_sub_dir = Path(self.checkpoint_dir) / self.model_run_name
        if not os.path.exists(self.checkpoint_sub_dir):
            rank_log(self.conf.is_main, self.logger.info, f"Creating new checkpoint dir at '{self.checkpoint_sub_dir}'.")
            if self.conf.is_main:
                os.makedirs(self.checkpoint_sub_dir, exist_ok=True)

    def compute_fitness(self, val_loss: float, mean_iou: float, gds: float) -> float:
        return self.w_miou * mean_iou + self.w_gds * gds - self.w_loss * val_loss

    def __call__(self, logs=None):
        """ Call the CheckpointManager to potentially save a checkpoint based on fitness score. """
        val_loss = logs.pop('val_loss', None)
        mean_iou = logs.pop('MeanIoU', None)
        gds = logs.pop('GeneralizedDiceScore', None)
        epoch = logs.pop('epoch', None)

        missing = [k for k, v in [('val_loss', val_loss), ('MeanIoU', mean_iou), ('GeneralizedDiceScore', gds)] if v is None]
        if missing:
            rank_log(self.conf.is_main, self.logger.warning, f"Metrics {missing} unavailable. Skipping checkpoint.")
            return None

        val_loss_f = val_loss.item() if isinstance(val_loss, torch.Tensor) else float(val_loss)
        mean_iou_f = mean_iou.item() if isinstance(mean_iou, torch.Tensor) else float(mean_iou)
        gds_f = gds.item() if isinstance(gds, torch.Tensor) else float(gds)

        fitness = self.compute_fitness(val_loss_f, mean_iou_f, gds_f)
        rank_log(
            self.conf.is_main, self.logger.info,
            f"Epoch {epoch} - fitness={fitness:.6f} (val_loss={val_loss_f:.6f}, MeanIoU={mean_iou_f:.6f}, GDS={gds_f:.6f})"
        )

        # Always save the most recent checkpoint
        if self.conf.is_main:
            config_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_config.yaml"
            if not os.path.exists(config_filename):
                OmegaConf.save(self.conf, config_filename)

            if len(self.most_recent_checkpoint) > 0:
                prev_chkpt_path = self.most_recent_checkpoint.pop()[1]
                if os.path.exists(prev_chkpt_path):
                    os.remove(prev_chkpt_path)
                    rank_log(self.conf.is_main, self.logger.info, f"Removed previous most recent checkpoint: {prev_chkpt_path}")
            chkpt_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_latest_epoch_{epoch}_fitness-{fitness:.6f}.pth"

            torch.save(logs, chkpt_filename)
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} checkpoint saved as most recent checkpoint. Saved to {chkpt_filename}")
            self.most_recent_checkpoint.append((epoch, chkpt_filename))

        # Save if top_k not yet full or this epoch has higher fitness than the current worst
        should_save = len(self.top_checkpoints) < self.top_k or fitness > self.top_checkpoints[-1][0]
        if should_save and self.conf.is_main:
            chkpt_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_epoch_{epoch}_fitness-{fitness:.6f}.pth"

            torch.save(logs, chkpt_filename)
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} - fitness improved or is in top-{self.top_k}. Saved to {chkpt_filename}")

            self.top_checkpoints.append((fitness, chkpt_filename))
            self.top_checkpoints.sort(key=lambda x: x[0], reverse=True)  # descending: best first, worst last

            if len(self.top_checkpoints) > self.top_k:
                worst_fitness, worst_path = self.top_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    rank_log(self.conf.is_main, self.logger.info, f"Removed checkpoint: {worst_path} with fitness={worst_fitness:.6f} (no longer in top-{self.top_k})")
        else:
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} - fitness did not improve top-{self.top_k}. Skipping checkpoint.")
