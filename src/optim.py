"""
src.optim.py
Optimizer and EMA classes
BoMeyering 2025
"""

import logging
import torch
import inspect
import argparse
from omegaconf import OmegaConf
from typing import List, Generator
import segmentation_models_pytorch as smp
from src.utils.loggers import rank_log

logger = logging.getLogger()

class OptimizerConfig:
    def __init__(self, conf, model_parameters):
        self.conf = conf
        self.optim_params = vars(self.conf.optimizer).copy()
        self.scheduler_params = vars(self.conf.scheduler).copy()
        self.model_params = model_parameters

    def get_optimizer(self):
        
        try:
            OptimClass = getattr(torch.optim, self.optim_params['name'])
        except AttributeError:
            rank_log(self.conf.local_rank, logger.warning, f"The loss function {self.optim_params['name']} is not in torch.nn. Defaulting to torch.optim.SGD.")
            OptimClass = torch.optim.SGD
            
        valid_params = inspect.signature(OptimClass).parameters
        filtered_params = {k: v for k, v in self.optim_params.items() if k in valid_params}

        optim_params = {'params': self.model_params}
        optim_params.update(filtered_params)

        optimizer = OptimClass(**optim_params)
        self.optimizer = optimizer
        return self.optimizer

    def get_scheduler(self):

        if self.args.scheduler.name:
            try:
                SchedClass = getattr(torch.optim.lr_scheduler, self.args.scheduler.name)
            except AttributeError:
                rank_log(self.conf.local_rank, logger.warning, f"The scheduler {self.args.scheduler.name} is not in torch.optim.lr_scheduler. Defaulting to torch.optim.")
                SchedClass = torch.optim.lr_scheduler.LinearLR

        valid_params = inspect.signature(SchedClass).parameters
        filtered_params = {k: v for k, v in self.scheduler_params.items() if k in valid_params}
        scheduler_params = {'optimizer': self.optimizer}
        scheduler_params.update(filtered_params)
        scheduler = SchedClass(**scheduler_params)

        self.scheduler = scheduler

        return self.scheduler

class EMA:
    def __init__(self, model: torch.nn.Module, decay: float, verbose: bool=True):
        """
        Initialize exponential moving average of named model parameters
        Smooths the model parameters based on an exponential moving average equation:

        s_(t) = decay * s_(t-1) + (1 - decay) * x_(t)

        Where:
        s_(t) = the shadow parameter value at time t
        decay = the EMA decay rate
        s_(t-1) = the shadow parameter value at time t-1
        x_(t) = the model parameter value at time t

        The higher the decay rate, the smoother the updates to the parameters are since they take more of the previous parameters values into account.

        Args:
            model (torch.nn.Module): The model to update
            decay (float): Exponential moving average decay rate
        """
        
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.original_params = {}
        self.update_params()
        self.verbose = verbose

    def  update_params(self):
        """
        Assigns the current parameters to the shadow params if they don't exist.
        Or updates the shadow_params by the decay rate and the current param values.
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow_params:
                    self.shadow_params[name] = param.data.clone()
                else:
                    self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data
    
    def assign_params(self):
        """
        Assign shadow parameters to the model's parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Update the original parameters in the original parameters dictionary
                self.original_params[name] = param.clone()
                # Copy the data from the shadow params to the model param
                param.data.copy_(self.shadow_params[name].data)

    def update(self):
        """
        Update shadow parameters and apply them to the model
        """
        self.update_params()
        self.assign_params()
        if self.verbose:
            print("Model parameters updated with shadow params")

    def restore_params(self):
        """
        Restores the original model parameters to the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_params[name].data)

