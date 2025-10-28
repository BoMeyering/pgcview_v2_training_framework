"""
src.callbacks.py
Callback Functions
BoMeyering 2025
"""

import torch
import os
import logging
from pathlib import Path
from datetime import datetime

class CheckpointManager:
    def __init__(self, checkpoint_dir: str='checkpoints', model_run_name: str='standard_run', monitor: str='val_loss', top_k: int=5, metadata=None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.model_run_name = model_run_name
        self.monitor = monitor
        self.monitor_op = torch.lt  # assume lower is better (e.g. val_loss)
        self.logger = logging.getLogger()
        self.metadata = metadata if metadata is not None else {}
        self.top_k = top_k
        self.top_checkpoints = []  # min-heap of (val_loss, filepath)

        if not os.path.exists(self.checkpoint_dir):
            self.logger.info(f"Creating new checkpoint dir at '{self.checkpoint_dir}'.")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def __call__(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            self.logger.warning(f"Warning: Metric '{self.monitor}' is not available. Skipping checkpoint.")
            return None

        # If we don't have enough checkpoints yet or current is better than the worst of top_k
        should_save = len(self.top_checkpoints) < self.top_k or self.monitor_op(current, self.top_checkpoints[-1][0])
        if should_save:
            # now = datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")
            chkpt_filename = self.checkpoint_dir / f"{self.model_run_name}_epoch_{epoch}_vloss-{current:.6f}.pth"
            chkpt = {
                'model_state_dict': logs['model_state_dict'],
                'epoch': epoch,
                'monitor': self.monitor,
                self.monitor: current,
                **self.metadata
            }
            torch.save(chkpt, chkpt_filename)
            self.logger.info(f"Epoch {epoch} - '{self.monitor}' improved or is in top-{self.top_k}. Saved to {chkpt_filename}")

            self.top_checkpoints.append((current, chkpt_filename))
            self.top_checkpoints.sort(key=lambda x: x[0])  # sort by val_loss (ascending)

            # If we now have too many checkpoints, remove the worst
            if len(self.top_checkpoints) > self.top_k:
                worst_loss, worst_path = self.top_checkpoints.pop()
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    self.logger.info(f"Removed checkpoint: {worst_path} with {self.monitor}={worst_loss:.6f} (no longer in top-{self.top_k})")
        else:
            self.logger.info(f"Epoch {epoch} - '{self.monitor}' did not improve top-{self.top_k}. Skipping checkpoint.")