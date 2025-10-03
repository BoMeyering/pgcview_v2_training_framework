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
from contextlib import contextmanager
from typing import List, Tuple, Optional, Any

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
        self.optim_params = conf.optimizer
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
        weight_decay = float(self.optim_params.weight_decay) if self.optim_params.weight_decay is not None else 0.0
        filter_bias_and_bn = bool(self.optim_params.filter_bias_and_bn) if self.optim_params.filter_bias_and_bn is not None else False

        if weight_decay > 0.0 and filter_bias_and_bn:
            logger.info(f"Applying weight decay={weight_decay} to model parameters. Bias and norm parameters will not be decayed.")

            parameters = self._add_weight_decay(weight_decay)
            # Stash original weight decay and set to 0
            setattr(self.optim_params, 'original_weight_decay', self.optim_params.weight_decay)
            setattr(self.optim_params, 'weight_decay', 0.0) # Set weight decay to 0 since it's handled in parameter groups

            return parameters, 0.0
        else:
            logger.info(f"Applying weight decay={weight_decay} to all model parameters.")
            return list(self._iter_trainable_params()), weight_decay


    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get the optimizer instance based on configuration.

        Returns:
        --------
            optimizer : torch.optim.Optimizer
                The instantiated optimizer.
        """
        # Get model parameters and updated weight decay
        self.model_params, self.optim_params.weight_decay = self._get_params()

        try:
            OptimClass = getattr(torch.optim, self.optim_params.name)
            logger.info(f"Using optimizer class {self.optim_params.name} from torch.optim")
        except AttributeError:
            valid_optim = [attr for attr in dir(torch.optim) if not attr.startswith("_") and inspect.isclass(getattr(torch.optim, attr))]
            logger.warning(f"The optimizer {self.optim_params.name} is not in ```torch.optim```")
            logger.warning(f"Must be one of {valid_optim}")
            logger.info("Defaulting to torch.optim.SGD.")
            OptimClass = torch.optim.SGD
        
        # Grab the valid parameters for the optimizer class and filter
        valid_params = inspect.signature(OptimClass).parameters
        filtered_params = {k: v for k, v in self.optim_params.items() if k in valid_params}

        # Combine model parameters with filtered optimizer parameters
        optim_params = {'params': self.model_params}
        optim_params.update(filtered_params)

        # Instantiate the optimizer
        optimizer = OptimClass(**optim_params)
        logger.info(f"Instantiated optimizer {self.optim_params.name}")  
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
            logger.error(f"The LR scheduler {self.scheduler_params.name} is not in ```torch.optim.lr_scheduler```")
            logger.error(f"Must be one of {valid_sched}")
            logger.info("Defaulting to torch.optim.lr_scheduler.LinearLR.")
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
        self.original_params = {}
        self.update_params()
        self.verbose = verbose

    def update_params(self):
        """
        Assigns the current parameters to the shadow params if they don't exist.
        Or updates the shadow_params by the decay rate and the current param values.
        """

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow_params:
                    self.shadow_params[name] = param.data.clone()
                else: # Update shadow params and apply decay
                    self.shadow_params[name] = self.decay * self.shadow_params[name] + (1 - self.decay) * param.data
    
    def assign_params(self):
        """
        Assign shadow parameters to the model's parameters
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Update the original parameters in the original parameters dictionary
                self.original_params[name] = param.data.clone()
                # Copy the data from the shadow params to the model param
                param.data.copy_(self.shadow_params[name].data)

    def update_and_assign_params(self):
        """
        Update shadow parameters and apply them to the model
        """
        self.update_params()
        self.assign_params()
        if self.verbose:
            logger.info("Model 'live' parameters stashed and model updated with shadow params")

    def restore_params(self):
        """
        Restores the original model parameters to the model
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original_params[name].data)
        if self.verbose:
            logger.info("Model 'live' parameters restored from stashed params")

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