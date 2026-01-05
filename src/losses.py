"""
src.losses.py
Loss Functions for Supervised and Semi-Supervised Learning
BoMeyering 2025
"""

import torch
import numbers
import src
import json
import logging
import inspect
import numpy as np
import torch.nn.functional as F
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Union, Optional,Tuple, Any
from omegaconf import OmegaConf
from src.utils.config import LossCriterion
from src.utils.loggers import rank_log
from src.distributed import is_main_process
import torch.distributed as dist

logger = logging.getLogger()

def read_class_counts(filepath: Union[str, Path]=Path('metadata/class_sample_counts.json'), tau: float=0.25) -> Tuple[Any]:
    """Read in a class count JSON file

    Load the calculated class pixel counts and return a list of sample counts as well as normalized, inverse frequency count weights

    Parameters:
    -----------
        filepath : Union[str, Path]
            The path to the sample counts JSON file. Defaults to 'metadata/class_sample_counts.json'
        tau : Power to modulate the weights by. Defaults to 0.25
    
    Returns:
    --------
        samples : list
            The list of sample pixel counts ordered by class index
        weights : list
            The list of normalized, modulated, inverse class weights ordered by class index
    """
    # Check type
    if not isinstance(filepath, (str, Path)):
        raise ValueError(
            f"'filepath' must be a valid string or pathlib.Path object; got {type(filepath)} instead."
        )
    # Convert to Path
    filepath = Path(filepath)
    # Check that path exits
    if not filepath.exists():
        raise FileExistsError(
            f"Filepath {str(filepath)} does not exist. Please check path integrity."
        )
    # Check file extension
    if filepath.suffix.lower() != '.json':
        raise ValueError(
            "File must be a valid json file"
        )
    
    try:
        with open(filepath, 'r') as f:
            count_dict = json.load(f)
            samples = [val['pixel_count'] for key, val, in count_dict.items()]

            # Invert and balance weights
            # Raise inverse weights to power of tau
            # Center them so that mean(norm_w) = 1 and clip within ratio limits
            inv_w = 1 / np.array([x for x in samples])
            tau_w = inv_w.clip(min=1e-12)**tau # Raise weights to tau_th power
            norm_w = (tau_w / tau_w.mean()).clip(min=1/5, max=5.0).tolist()

            return samples, norm_w
        
    except JSONDecodeError as e:
        logger.error(f"Error decoding JSON file. Error: {e}")

        return None, None

def get_loss_criterion(conf: OmegaConf) -> torch.nn.Module:
    """
    Get a loss function from the configuration

    Parameters:
    -----------
        conf: omegaconf.OmegaConf
            The configuration dictionary from the config file.

    Returns:
    --------
        criterion : torch.nn.Module
            An instantiated loss criterion.
    """

    # Set loss name and retrieve the class from src.losses namespace
    loss_name = conf.loss.name.value
    LossClass = getattr(src.losses, loss_name)

    # Get the loss parameters from the config
    loss_params = conf.loss
    # Get the valid parameters for the loss class and filter the config params
    valid_params = inspect.signature(LossClass).parameters
    filtered_params = {k: v for k, v in loss_params.items() if k in valid_params}

    # Set the loss_type parameter to the Enum name if it exists
    if 'loss_type' in filtered_params:
        filtered_params['loss_type'] = LossCriterion.__members__.get(filtered_params['loss_type']).value
    # Convert samples to a torch tensor if it exists
    if 'samples' in filtered_params:
        filtered_params['samples'] = torch.tensor(filtered_params['samples'], dtype=torch.float32).to(conf.device)
    # Convert weights to a torch tensor if it exists
    if 'weights' in filtered_params:
        filtered_params['weights'] = torch.tensor(filtered_params['weights'], dtype=torch.float32).to(conf.device)

    # Instantiate the criterion
    criterion = LossClass(**filtered_params)
    
    return criterion

class CELoss(torch.nn.Module):
    """
    Wrapper class for vanilla cross entropy loss.
    """
    def __init__(
            self,
            smooth: float=0.0, 
            weights: Optional[torch.Tensor]=None, 
            reduction: str='mean'
        ):
        """Instantiate a CELoss object.

        Parameters:
        -----------
            smooth : float, optional
                A float in the range [0.0, 1.0]. Specifies the amount of smoothing to apply to the labels, by default 0.0
            weights : torch.Tensor, optional
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the weight for that class, by default None.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
        """
        super().__init__()
        
        # Validate arguments
        if not isinstance(smooth, float) or not (0.0 <= smooth <= 1.0):
            raise ValueError(
                "'smooth' must be a float in the range [0.0, 1.0]; got {smooth} instead."
            )
        self.smooth = smooth

        if not isinstance(weights, (torch.Tensor, type(None))):
            raise ValueError(
                f"'weights' must be a torch.Tensor or None; got {type(weights)} instead."
            )
        self.weights = weights

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction

    def forward(
        self, 
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Forward method of CELoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        
        if isinstance(self.weights, torch.Tensor) and len(self.weights) != logits.shape[1]:
            raise ValueError(
                f"Length of weights should be the number of classes in logits, {logits.shape[1]}. Please check the weights or the logits passed."
            )
        # Clone so we don’t modify caller’s tensor
        tgt = targets.clone()

        if mask is not None:
            tgt[~mask] = -1

        weight = self.weights if self.weights is not None else None

        loss = F.cross_entropy(
            logits,
            tgt,
            weight=weight,
            ignore_index=-1,
            reduction=self.reduction,
            label_smoothing=self.smooth,
        )
        return loss

class FocalLoss(torch.nn.Module):
    """
    Implementation of Focal Loss
    https://arxiv.org/abs/1708.02002
    """
    def __init__(
            self,
            smooth: float=0.0,
            weights: Optional[torch.Tensor]=None,
            reduction: str='mean',
            gamma: float=2.0
        ):
        """Instantiate a FocalLoss object.

        Parameters:
        -----------
            smooth : float, optional
                A float in the range [0.0, 1.0]. Specifies the amount of label smoothing to apply, by default 0.0.
            weights : torch.Tensor, optional
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the weight for that class, by default None
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum' , 'none']. Defaults to 'mean'.
            gamma : float, optional
                Focusing parameter for modulating factor (1-p). Defaults to 2.0.
        """
        super().__init__()
        self.eps = 1e-8
        
        # Validate arguments
        if not isinstance(smooth, float) or not (0.0 <= smooth <= 1.0):
            raise ValueError(
                "'smooth' must be a float in the range [0.0, 1.0]; got {smooth} instead."
            )
        self.smooth = smooth

        if not isinstance(weights, (torch.Tensor, type(None))):
            raise ValueError(
                f"'weights' must be a torch.Tensor or None; got {type(weights)} instead."
            )
        self.weights = weights

        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction

        if not isinstance(gamma, numbers.Real) or gamma < 0.0:
            raise ValueError(
                f"'gamma' must be a non-negative float; got {gamma} instead."
            )
        self.gamma = float(gamma)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """Forward method of FocalLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        N, C = logits.shape[:2]
        if isinstance(self.weights, torch.Tensor) and len(self.weights) != logits.shape[1]:
            raise ValueError(
                f"Length of weights should be the number of classes in logits, {logits.shape[1]}. Please check the weights or the logits passed."
            )
        
        if mask is None:
            mask = torch.ones_like(targets, dtype=torch.bool)

        if self.weights is not None:
            self.weights.to(logits.device, dtype=logits.dtype)
            shape = [1, -1] + [1]*(logits.ndim-2)
            class_weights = self.weights.view(*shape)
        else:
            class_weights = 1.0

        one_hot = F.one_hot(targets, num_classes=C).movedim(-1, 1).float()
        if self.smooth > 0.0:
            y_soft = (1 - self.smooth) * one_hot + (1 - one_hot) * self.smooth / (C - 1)
        else:
            y_soft = one_hot
        
        log_probs = torch.log_softmax(logits, dim=1)    # N, C, H, W
        probs = log_probs.exp()                         # N, C, H, W

        focal_factor = (1 - probs).pow(self.gamma)
        
        loss_per_class = - y_soft * class_weights * focal_factor * log_probs
        loss_per_pixel = loss_per_class.sum(dim=1) # N, H, W
        loss_per_pixel = loss_per_pixel * mask

        if self.reduction == 'none':
            return loss_per_pixel
        elif self.reduction == 'sum':
            return loss_per_pixel.sum()
        elif self.reduction == 'mean':
            return loss_per_pixel.sum() / mask.sum().clamp(min=1)

class CBLoss(torch.nn.Module):
    """
    Implementation of Class Balanced losses
    https://arxiv.org/pdf/1901.05555
    """
    def __init__(
            self, 
            samples: torch.Tensor, 
            loss_type: str='CELoss', 
            smooth: float=0.0,
            reduction: str='mean', 
            gamma: float=2.0
        ):
        """Instantiate a CBLoss object.

        Parameters:
        -----------
            samples : torch.Tensor
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
            smooth : float, optional
                A float in the range [0.0, 1.0]. Specifies the amount of smoothing to apply to the labels, by default 0.0.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum']. Defaults to 'mean'.
            gamma : float, optional
                The gamma value to use for Focal Loss. Only used if loss_type is 'FocalLoss'. Defaults to 2.0.

        Raises:
        -------
            ValueError: If loss_type is not one of ['CELoss', 'FocalLoss'].
        """
        super().__init__()
        self.samples = samples.to(dtype=torch.float64)
        self.loss_type = loss_type
        self.smooth = smooth
        self.reduction = reduction
        self.gamma = gamma

        # Calculate effective number of samples
        self.N = self.samples.sum().to(dtype=torch.float64)
        self.beta = (self.samples - 1) / self.samples
        self.C = len(self.samples)
        self.eps = 1e-8

        # Set the loss function with the proper weights
        self._set_effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(smooth=self.smooth, weights=self.weights, reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(smooth=self.smooth, weights=self.weights, reduction=self.reduction, gamma=self.gamma)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")
        
    def _tau_search(self, wmin: float, wmax: float, spread: float) -> float:
        """ Bisection search to find tau given spread constraint """
        diff = np.inf
        tol = 1e-8
        tl = 0.0
        tu = 5.0
        current = 1000

        while diff > tol:
            tm = (tl + tu) / 2

            ftm = wmax**tm - wmin**tm - spread

            if ftm > 0:
                tu = tm
            elif ftm < 0:
                tl = tm

            diff = np.abs(tm - current)
            if diff < tol:
                break
            else:
                current = tm
        
        return tm
    
    def _set_effective_samples(self):
        """Helper function to calculate the effective samples and weights.
        
        Effective samples E_n = (1 - beta^n) / (1 - beta)
        Weights weights_n = C / (E_n * sum(1/E_n)) such that sum(weights_n) = C
        """

        # Calculate effective samples
        E = (1.0 - self.beta ** self.samples) / (1.0 - self.beta + self.eps)

        # Invert to get weights weights and normalize to sum to C
        invE = 1.0 / E
        weights = invE / invE.mean()

        # Modulate by tau and then do one final normalization
        tau = self._tau_search(wmin=weights.min().item(), wmax=weights.max().item(), spread=2.0)
        invE = weights.clamp(min=1e-12)**tau
        weights = invE / invE.mean()

        self.weights = weights.to(dtype=torch.float32, device=self.samples.device)

        if not torch.isfinite(self.weights).all():
            raise ValueError(
                "Non-finite class balanced weights computed. Check implementation in src.losses.CBLoss"
            )
     
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """Forward method of CBLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        loss = self.loss_fn(logits=logits, targets=targets, mask=mask)

        return loss
  
class ACBLoss(torch.nn.Module):
    """
    Implement Adaptive Class Balanced Loss from Xu et al 2022.
    https://ieeexplore.ieee.org/document/10137858
    """
    def __init__(
            self, 
            samples: torch.Tensor, 
            loss_type: str, 
            smooth: float=0.0,
            reduction: str='mean', 
            gamma: float=2.0
        ):
        """Instantiate an ACBLoss object.

        Parameters:
        -----------
            samples : torch.Tensor
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
            smooth : float, optional
                A float in the range [0.0, 1.0]. Specifies the amount of smoothing to apply to the labels, by default 0.0
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum']. Defaults to 'mean'.
            gamma : float, optional
                The gamma value to use for Focal Loss. Only used if loss_type is 'FocalLoss'. Defaults to 2.0.

        Raises:
        -------
            ValueError: If loss_type is not one of ['CELoss', 'FocalLoss'].
        """
        super().__init__()
        self.samples = samples.double()
        self.loss_type = loss_type
        self.smooth = smooth
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.N_max = torch.max(self.samples)
        self.C = len(self.samples)
        self.eps = 1e-8

        self._set_effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(smooth=self.smooth, weights=self.weights.float(), reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(smooth=self.smooth, weights=self.weights.float(), reduction=self.reduction, gamma=self.gamma)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _set_effective_samples(self):
        """Helper function to calculate the effective samples and weights based on beta.

        Beta = F(f(u, v, b)) = tanh(u / (v * sqrt(b)))
        where u = log(N), v = log(C), b = -mean(log10(n_i / N_max))
        F is the squashing function tanh to ensure beta is in [0, 1)
        n_i is the number of samples for class i, and N_max is the maximum number of samples in any class. 
        E_n = (1 - beta^n) / (1 - beta)
        weights_n = C / (E_n * sum(1/E_n)) such that sum(weights_n) = C
        """
        # Sample size, class size, and degree of imbalance calculations
        self.u = torch.log(self.N.double())
        self.v = torch.log(torch.tensor(self.C).double())
        self.b = -torch.log10(self.samples / self.N_max).mean().double()
        self.f_uvb = self.u / (self.v ** torch.sqrt(self.b)).double()
        self.beta = torch.tanh(self.f_uvb).double()

        # Calculate effective samples
        E = (1 - torch.pow(self.beta, self.samples)).double() / (1 - self.beta + self.eps).double()

        # Invert to get weights weights and normalize
        invE = 1.0 / E
        weights = invE / invE.mean()

        self.E = E
        self.weights = weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Forward method of ACBLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        
        loss = self.loss_fn(logits=logits, targets=targets, mask=mask)
        
        return loss

class RecallLoss(torch.nn.Module):
    """
    Implementation of Recall Loss with dynamic weighting
    https://arxiv.org/pdf/2106.14917
    """
    def __init__(
            self, 
            samples: torch.Tensor, 
            loss_type: str, 
            smooth: float=0.0,
            reduction: str='mean', 
            gamma: float=2.0, 
            eps: float=1e-8
        ):
        """Instantiate a RecallLoss object.

        Parameters:
        -----------
            samples : torch.Tensor 
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum']. Defaults to 'mean'.
            gamma : float, optional
                The gamma value to use for Focal Loss. Only used if loss_type is 'FocalLoss'. Defaults to 2.0.
            eps : float, optional
                A small value to avoid division by zero. Defaults to 1e-8.
        """
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.smooth = smooth
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.C = len(self.samples)
        self.eps = eps

        # Set the loss function with the proper weights
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(smooth=self.smooth, weights=None, reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(smooth=self.smooth, weights=None, reduction=self.reduction, gamma=self.gamma)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _calculate_weights(self, logits: torch.Tensor, targets: torch.Tensor):
        """Helper function to calculate the recall weights.
        
        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
        """
        # Get probs from logits and calculate one-hot tensors
        probs = F.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1)
        pred_oh = F.one_hot(pred_labels, num_classes=self.C)
        target_oh = F.one_hot(targets, num_classes=self.C)

        # Reshape to (-1, num_classes) to sum over one dimension
        pred_oh = pred_oh.view(-1, self.C)
        target_oh = target_oh.view(-1, self.C)

        # Calculate TP and FN rates
        TP = ((target_oh == 1) * (pred_oh == 1)).sum(dim=0)
        FN = ((target_oh == 1) * (pred_oh == 0)).sum(dim=0)

        # Calculate recall and weights
        R_c = (TP / (FN + TP + self.eps)).clamp(min=self.eps)
        weights = 1 - R_c
        
        # Set weights to uniform if all weights are zero
        if torch.all(weights == 0):
            weights = torch.full_like(weights, 1.)

        # Normalize weights to sum to C
        self.weights = weights / weights.mean()
        # Update the loss_fn with the new weights
        self.loss_fn.weights = self.weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """
        Forward method of RecallLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        # Calculate effective samples
        self._calculate_weights(logits=logits, targets=targets)
        
        # Compute the loss
        loss = self.loss_fn(logits=logits, targets=targets, mask=mask)
        
        return loss

class DiceLoss(torch.nn.Module):
    """
    Implementation of Dice Loss
    https://arxiv.org/abs/1606.04797
    """
    def __init__(
            self, 
            reduction: str = 'mean',
            gamma: Optional[float]=1.0,
        ):
        """Instantiate a DiceLoss object.

        Parameters:
        -----------
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
            gamma : float, optional
                An optional float focusing parameter for a focal variant of DiceLoss
        """
        super().__init__()
        self.gamma = gamma
        self.eps = 1.0 # Constant to avoid zero division

        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None, return_stats: bool=False) -> torch.Tensor:
        """Forward method of DiceLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (C,).
        """
        # 
        probs = torch.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets, num_classes=probs.shape[1]).movedim(-1, 1).float()
        reduce_dims = tuple(d for d in range(probs.ndim) if d not in (1,))
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            probs = probs * mask
            targets_oh = targets_oh * mask
        
        # Sum operations only affect non-masked pixels so no special handling needed during reduction
        intersection = (probs * targets_oh).sum(reduce_dims)
        probs_2 = (probs ** 2).sum(dim=reduce_dims)
        gt_2 = (targets_oh ** 2).sum(dim=reduce_dims)

        dc_per_class = (2.0 * intersection + self.eps) / (probs_2 + gt_2 + self.eps)
        
        # Compute the loss and apply focal power
        # When focal_gamma == 1, this is the same a unfocused Dice Loss
        loss = (1.0 - dc_per_class).clamp(min=0.0).pow(self.gamma)

        # Return stats loop
        if return_stats:
            if self.reduction == 'mean':
                return loss.mean(), intersection, probs_2 + gt_2
            elif self.reduction == 'sum':
                return loss.sum(), intersection, probs_2 + gt_2
            else:
                return loss, intersection, probs_2 + gt_2
        else:
            if self.reduction == 'mean':
                return loss.mean()
            elif self.reduction == 'sum':
                return loss.sum()
            else:
                return loss

class TverskyLoss(torch.nn.Module):
    """
    Implementation of Tversky Loss
    https://arxiv.org/abs/1706.05721
    """
    def __init__(
            self, 
            alpha: float = 0.5, 
            beta: float = 0.5, 
            reduction: str = 'mean'
        ):
        """Instantiate a TverskyLoss object.

        Parameters:
        -----------
            alpha : float, optional
                Weight for false negatives. Defaults to 0.5.
            beta : float, optional
                Weight for false positives. Defaults to 0.5.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction
        self.eps = 1.0
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """Forward method of TverskyLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W). Do not pass Softmax probabilities.
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss : torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        # Get probabilities from logits and convert targets to one-hot
        probs = torch.softmax(logits, dim=1).clamp(min=self.eps, max=1-self.eps) # Clamp the logits in the inverval [self.eps, 1-self.eps]
        targets_one_hot = F.one_hot(targets, num_classes=probs.shape[1]).movedim(-1, 1).float()
        reduce_dims = tuple(d for d in range(probs.ndim) if d not in (1,))
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # Calculate true positives, false negatives, and false positives based on the softmax probabilities
        true_pos = (probs * targets_one_hot).sum(dim=reduce_dims)
        false_neg = ((1 - probs) * targets_one_hot).sum(dim=reduce_dims)
        false_pos = (probs * (1 - targets_one_hot)).sum(dim=reduce_dims)

        # Calculate Tversky index and loss per class
        tversky_index = (true_pos + self.eps) / (true_pos + self.alpha * false_neg + self.beta * false_pos + self.eps)
        loss = 1 - tversky_index

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TvmfDiceLoss(torch.nn.Module):
    """
    Implementation of t-vMF Dice Loss
    https://www.sciencedirect.com/science/article/pii/S0010482523011605#fig1
    """
    def __init__(
        self, 
        kappa: Optional[float]=0,
        smooth: float = 0.0, 
        reduction: str = 'mean',
        exclude_empty_target: bool=True,
        lambda_k: Optional[float]=None
    ):
        """Instantiate a TvmfDiceLoss object.

        Parameters:
        -----------
            kappa : float, optional
                Weights the denominator of the t-vMF loss
            smooth : float, optional
                A smoothing factor to avoid division by zero. Defaults to 0.0.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
            exclude_empty_target : bool, optional
                Whether to exclude classes with no target pixels from the loss calculation. Defaults to True.
            lambda_k : float, optional
                Float value used to determine adaptive kappa. Either float or None. Defaults to None.
        """
        super().__init__()
        self.kappa = kappa
        self.smooth = smooth
        self.reduction = reduction
        self.eps = 1e-8
        self.exclude_empty_target = bool(exclude_empty_target)
        self.lambda_k = lambda_k



    def _flatten_per_class(self, X: torch.Tensor) -> torch.Tensor:
        X = X.movedim(1, 0).contiguous()

        return X.reshape(X.shape[0], -1)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        """Forward method of TvmfDiceLoss.

        Parameters:
        -----------
            logits : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to include. Defaults to None.

        Returns:
        --------
            loss :  torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (C,).
        """
        # Calculate probabilities and targets
        probs = torch.softmax(logits, dim=1)
        N, C = probs.shape[:2]
        targets_oh = F.one_hot(targets, num_classes=C).movedim(-1, 1).float()

        # Flatten and normalize vectors for cosine similarity
        A = self._flatten_per_class(probs) # Tensor of shape (C, N*H*W)
        B_raw = self._flatten_per_class(targets_oh)

        if mask is not None:
            mask = mask.reshape(1, -1).expand(C, -1)
            A = A[mask].reshape(C, -1)
            B_raw = B_raw[mask].reshape(C, -1)

        B = B_raw
        # Apply label smoothing if specified
        if self.smooth > 0.0:
            B = B * (1.0 - self.smooth) + (1.0 - B) * self.smooth / (C - 1)

        A = A / (A.norm(dim=1, keepdim=True) + self.eps)
        B = B / (B.norm(dim=1, keepdim=True) + self.eps)

        # Calculate cosine similarity between the vectors and clamp
        cos_theta = (A * B).sum(dim=1) # Sum over N*H*W vector per class -> tensor of shape (C,)
        cos_theta = cos_theta.clamp(-1.0 + self.eps, 1.0 - self.eps)

        phi_t = (1.0 + cos_theta) / (1.0 + self.kappa*(1.0 - cos_theta)) - 1.0

        class_loss = (1.0 - phi_t) ** 2

        if self.exclude_empty_target:
            idx = B_raw.norm(dim=1) > 0
            if idx.any():
                class_loss = class_loss[idx]
            else:
                return logits.new_tensor(0.0)

        if self.reduction == 'mean':
            loss = torch.mean(class_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(class_loss)
        elif self.reduction == 'none':
            loss = class_loss

        # Update adaptive kappa if lambda_k is set
        if self.lambda_k is not None:
            dsc_t = torch.mean(class_loss.detach())
            kappa = dsc_t * self.lambda_k
            dist.all_reduce(kappa, op=dist.ReduceOp.SUM)
            kappa /= dist.get_world_size()
            self.kappa = kappa.item()
            
        return loss
    