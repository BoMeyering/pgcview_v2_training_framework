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
    def __init__(self, conf: OmegaConf, monitor: str='val_loss', monitor_op=torch.lt, top_k: int=5):
        self.checkpoint_dir = Path(conf.directories.checkpoint_dir)
        self.model_run_name = conf.model_run
        self.monitor = monitor
        self.monitor_op = monitor_op  # assume lower is better (e.g. val_loss)
        self.logger = logging.getLogger()
        self.conf = conf
        self.top_k = top_k
        self.top_checkpoints = []  # min-heap of (val_loss, filepath)
        self.most_recent_checkpoint = []

        self.checkpoint_sub_dir = Path(self.checkpoint_dir) / self.model_run_name
        if not os.path.exists(self.checkpoint_sub_dir):
            rank_log(self.conf.is_main, self.logger.info, f"Creating new checkpoint dir at '{self.checkpoint_sub_dir}'.")
            if self.conf.is_main:
                os.makedirs(self.checkpoint_sub_dir, exist_ok=True)

    def __call__(self, logs=None):
        """ Call the CheckpointManager to potentially save a checkpoint based on monitored metric. """
        current = logs.pop(self.monitor, None)
        epoch = logs.pop('epoch', None)

        if current is None:
            rank_log(self.conf.is_main, self.logger.warning, f"Warning: Metric '{self.monitor}' is not available. Skipping checkpoint.")
            return None

        # Always save the most recent checkpoint
        if self.conf.is_main:
            config_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_config.yaml"
            if not os.path.exists(config_filename):
                OmegaConf.save(self.conf, config_filename)

            if len(self.most_recent_checkpoint) > 0:
                # Remove previous most recent checkpoint
                prev_chkpt_path = self.most_recent_checkpoint.pop()[1]
                if os.path.exists(prev_chkpt_path):
                    os.remove(prev_chkpt_path)
                    rank_log(self.conf.is_main, self.logger.info, f"Removed previous most recent checkpoint: {prev_chkpt_path}")
            chkpt_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_latest_epoch_{epoch}_vloss-{current:.6f}.pth"

            # Form the checkpoint dict
            # chkpt = {
            #     'model_state_dict': logs.get('model_state_dict'),
            #     'ema_state_dict': logs.get('ema_state_dict'),
            # }

            torch.save(logs, chkpt_filename)
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} checkpoint saved as most recent checkpoint. Saved to {chkpt_filename}")

            self.most_recent_checkpoint.append((epoch, chkpt_filename))

        # If we don't have enough checkpoints yet or current is better than the worst of top_k
        should_save = len(self.top_checkpoints) < self.top_k or self.monitor_op(current, self.top_checkpoints[-1][0])
        if should_save and self.conf.is_main:
            chkpt_filename = self.checkpoint_sub_dir / f"{self.model_run_name}_epoch_{epoch}_vloss-{current:.6f}.pth"

            # Form the checkpoint dict
            # chkpt = {
            #     'model_state_dict': logs.get('model_state_dict'),
            #     'ema_state_dict': logs.get('ema_state_dict'),
            # }

            torch.save(logs, chkpt_filename)
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} - '{self.monitor}' improved or is in top-{self.top_k}. Saved to {chkpt_filename}")

            self.top_checkpoints.append((current, chkpt_filename))
            self.top_checkpoints.sort(key=lambda x: x[0])  # sort by val_loss (ascending)

            # If we now have too many checkpoints, remove the worst
            if len(self.top_checkpoints) > self.top_k:
                worst_loss, worst_path = self.top_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    rank_log(self.conf.is_main, self.logger.info, f"Removed checkpoint: {worst_path} with {self.monitor}={worst_loss:.6f} (no longer in top-{self.top_k})")
        else:
            rank_log(self.conf.is_main, self.logger.info, f"Epoch {epoch} - '{self.monitor}' did not improve top-{self.top_k}. Skipping checkpoint.")
