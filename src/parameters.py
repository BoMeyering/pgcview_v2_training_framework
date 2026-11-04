"""
src/parameters.py
Model Parameters and Exponential Moving Average
BoMeyering 2025
"""

import torch
import omegaconf
import inspect
import logging
import torch.nn as nn
import torch.distributed as dist
from contextlib import contextmanager
from typing import List, Tuple, Optional, Any
from src.utils.loggers import rank_log
from src.distributed import is_main_process


# Set up logger
logger = logging.getLogger()

#--------------------------------#
# Optimizer Configuration Class
#--------------------------------#

NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm, nn.GroupNorm, nn.InstanceNorm1d,
    nn.InstanceNorm2d, nn.InstanceNorm3d
)

class OptimConfig:
    """
    Class to handle optimizer and scheduler configuration and instantiation.
    """
    def __init__(
            self, 
            conf: omegaconf.OmegaConf,
            model: torch.nn.Module,
        ):
        """Instantiate the OptimConfig class.

        Parameters:
        -----------
            conf: omegaconf.OmegaConf
                The configuration object containing optimizer and scheduler parameters.
            model: torch.nn.Module
                The model containing parameters to be optimized.
        """
        if not isinstance(conf, omegaconf.dictconfig.DictConfig):
            raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
        self.conf = conf
        self.model = model
        self.optim_args = conf.optimizer
        self.optim_params = conf.optimizer.optimizer_params
        self.scheduler_params = conf.scheduler
    
    def _add_weight_decay(
            self,
            weight_decay: float=1e-5
        ) -> List[dict[str, Any]]:
        """Returns parameter groups with and without weight decay.

        Parameters:
        -----------
            weight_decay : float, optional
                The weight decay factor to be applied to applicable parameters. Default is 1e-5

        Returns:
        --------
            parameter_groups : List[dict]
                A list of parameter groups for the optimizer.
        """

        # Set up parameter groups
        decay = []
        no_decay = []
        for module in self.model.modules():
            # Norm layers - don't apply weight decay
            if isinstance(module, NORM_TYPES):
                for p in module.parameters(recurse=False):
                    if p.requires_grad:
                        no_decay.append(p)
                continue

            # Other layers: bias etc - don't apply weight decay
            for name, p in module.named_parameters(recurse=False):
                if not p.requires_grad:
                    continue
                if name == "bias":
                    no_decay.append(p)
                else:
                    decay.append(p)
        
        parameter_groups = []
        if no_decay:
            parameter_groups.append({'params': no_decay, 'weight_decay': 0.0})
        if decay:
            parameter_groups.append({'params': decay, 'weight_decay': float(weight_decay)})
        
        return parameter_groups
    
    def _iter_trainable_params(self):
        """ 
        Generator to iterate over trainable model parameters.
        """
        for p in self.model.parameters():
            if p.requires_grad:
                yield p
    
    def _get_params(self) -> Tuple:
        """Returns model parameters and updated weight decay value based on args.
        
        Returns:
        -------
            Tuple: A tuple containing the model parameters and the udpated weight decay value for the optimizer
        """
        weight_decay = float(self.optim_args.weight_decay) if self.optim_args.weight_decay is not None else 0.0
        filter_bias_and_bn = bool(self.optim_args.filter_bias_and_bn) if self.optim_args.filter_bias_and_bn is not None else False

        if weight_decay > 0.0 and filter_bias_and_bn:
            rank_log(self.conf.is_main, logger.info, f"Applying weight decay={weight_decay} to model parameters. Bias and norm parameters will not be decayed.")

            parameters = self._add_weight_decay(weight_decay)
            # Stash original weight decay and set to 0
            setattr(self.optim_args, 'original_weight_decay', self.optim_args.weight_decay)
            setattr(self.optim_args, 'weight_decay', 0.0) # Set weight decay to 0 since it's handled in parameter groups

            return parameters, 0.0
        else:
            rank_log(self.conf.is_main, logger.info, f"Applying weight decay={weight_decay} to all model parameters.")
            return list(self._iter_trainable_params()), weight_decay


    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer instance based on configuration.

        Returns:
        --------
            optimizer : torch.optim.Optimizer
                The instantiated optimizer.
        """
        # Get model parameters and updated weight decay
        self.model_params, self.optim_args.weight_decay = self._get_params()

        try:
            OptimClass = getattr(torch.optim, self.optim_args.name)
            rank_log(self.conf.is_main, logger.info, f"Using optimizer class {self.optim_args.name} from torch.optim")
        except AttributeError:
            valid_optim = [attr for attr in dir(torch.optim) if not attr.startswith("_") and inspect.isclass(getattr(torch.optim, attr))]
            rank_log(self.conf.is_main, logger.warning, f"The optimizer {self.optim_args.name} is not in ```torch.optim```")
            rank_log(self.conf.is_main, logger.warning, f"Must be one of {valid_optim}")
            rank_log(self.conf.is_main, logger.info, "Defaulting to torch.optim.SGD.")
            OptimClass = torch.optim.SGD
        
        # Grab the valid parameters for the optimizer class and filter
        valid_params = inspect.signature(OptimClass).parameters
        filtered_params = {k: v for k, v in self.optim_params.items() if k in valid_params}

        # Combine model parameters with filtered optimizer parameters
        optim_params = {'params': self.model_params}
        optim_params.update(filtered_params)

        # Instantiate the optimizer
        optimizer = OptimClass(**optim_params)
        rank_log(self.conf.is_main, logger.info, f"Instantiated optimizer {self.optim_args.name}")
        self.optimizer = optimizer

        return optimizer

    def _get_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Get the learning rate scheduler instance based on configuration.

        Returns:
        --------
            scheduler : torch.optim.lr_scheduler._LRScheduler
                The instantiated learning rate scheduler.
        """
        #
        try:
            SchedClass = getattr(torch.optim.lr_scheduler, self.scheduler_params.name)
        except AttributeError:
            valid_sched = [
                attr for attr in dir(torch.optim.lr_scheduler)
                if attr.__contains__("LR")
                and not attr.startswith("_")
                and inspect.isclass(getattr(torch.optim.lr_scheduler, attr))
            ]
            rank_log(self.conf.is_main, logger.error, f"The LR scheduler {self.scheduler_params.name} is not in ```torch.optim.lr_scheduler```")
            rank_log(self.conf.is_main, logger.error, f"Must be one of {valid_sched}")
            rank_log(self.conf.is_main, logger.info, "Defaulting to torch.optim.lr_scheduler.LinearLR.")

            SchedClass = torch.optim.lr_scheduler.LinearLR

        # Grab the valid parameters for the scheduler class and filter
        valid_params = inspect.signature(SchedClass).parameters
        filtered_params = {k: v for k, v in self.scheduler_params.items() if k in valid_params}

        # Combine optimizer with filtered scheduler parameters
        scheduler_params = {'optimizer': self.optimizer}
        scheduler_params.update(filtered_params)

        # Instantiate the scheduler
        scheduler = SchedClass(**scheduler_params)
        self.scheduler = scheduler

        return scheduler
    
    def process(self) -> Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Process the configuration to create optimizer and scheduler instances.

        Returns:
        --------
            Tuple[torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
                A tuple containing the instantiated optimizer and scheduler.
        """
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler()
        model = self.model

        return model, optimizer, scheduler


class EMA:
    """
    Implements Exponential Moving Average (EMA) for model parameters.
    """
    def __init__(
            self, 
            model: torch.nn.Module, 
            decay: float=.9, 
            verbose: bool=True
        ):
        """Instantiate the EMA class.

        Initialize exponential moving average of named model parameters
        Smooths the model parameters based on an exponential moving average equation:

        s_(t) = decay * s_(t-1) + (1 - decay) * x_(t)

        Where:
            s_(t) = the shadow parameter value at time t
            decay = the EMA decay rate
            s_(t-1) = the shadow parameter value at time t-1
            x_(t) = the model parameter value at time t

        The higher the decay rate, the smoother the updates to the parameters are since they take more of the previous parameters values into account.

        Parameters:
        -----------
            model : torch.nn.Module
                The model to update
            decay : float
                Exponential moving average decay rate
        """
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self._orig_params_cpu = {}
        self.update_params()
        self.verbose = verbose
        
    @torch.no_grad()
    def update_params(self):
        """
        Assigns the current parameters to the shadow params if they don't exist.
        Or updates the shadow_params by the decay rate and the current param values.
        """

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self.shadow_params:
                self.shadow_params[name] = param.detach().to('cpu').clone()
            else:
                self.shadow_params[name].mul_(self.decay).add_(
                    param.detach().to('cpu'), alpha=1-self.decay
                )
                    
    @torch.no_grad()
    def assign_params(self):
        """
        Assign shadow parameters to the model's parameters
        """
        self._orig_params_cpu.clear()
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Stash original param on cpu
            self._orig_params_cpu[name] = param.detach().to('cpu').clone()

            # Copy EMA param to live param
            param.copy_(self.shadow_params[name].to(param.device, non_blocking=True))

    def update_and_assign_params(self):
        """
        Update shadow parameters and apply them to the model
        """
        self.update_params()
        self.assign_params()

    @torch.no_grad()
    def restore_params(self):
        """
        Restores the original model parameters to the model
        """
        # Move original params back to live model params
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            param.copy_(self._orig_params_cpu[name].to(param.device, non_blocking=True))
        
        # Clear out the cpu params dict
        self._orig_params_cpu.clear()

@contextmanager
def apply_ema(ema):
    """Context manager to apply EMA parameters during evaluation and restore original parameters afterwards.

    Parameters:
    -----------
        ema : src.parameters.EMA
            The EMA instance containing shadow and original parameters.
    """
    ema.assign_params()
    try:
        yield
    finally:
        ema.restore_params()