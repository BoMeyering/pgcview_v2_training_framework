"""
src.losses.py
Loss Functions for Supervised and Semi-Supervised Learning
BoMeyering 2025
"""

import torch
import src
import json
import logging
import inspect
import argparse
import torch.nn.functional as F
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Union, Optional
from omegaconf import OmegaConf
from src.utils.config import LossCriterion
from src.utils.loggers import rank_log
from src.distributed import is_main_process

logger = logging.getLogger()

def read_class_counts(filepath: Union[str, Path]=Path('metadata/class_sample_counts.json')):
    """Read in a class count JSON file

    Load the calculated class pixel counts and return a list of sample counts as well as inverse weights

    Parameters:
    -----------
        filepath : Union[str, Path]
            The path to the sample counts JSON file. Defaults to 'metadata/class_sample_counts.json'
    
    Returns:
    --------
        samples : list
            The list of sample pixel counts ordered by class index
        inv_weights : list
            The list of inverse class weights ordered by class index
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
        print(filepath.suffix.lower())
        raise ValueError(
            "File must be a valid json file"
        )
    
    try:
        with open(filepath, 'r') as f:
            count_dict = json.load(f)
            samples = [val['pixel_count'] for key, val, in count_dict.items()]
            inv_weights = [1/(x/sum(samples)) for x in samples]

            return samples, inv_weights
        
    except JSONDecodeError as e:
        rank_log(IS_MAIN, logger.error, f"Error decoding JSON file. Error: {e}")

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
    # Ge the valid parameters for the loss class and filter the config params
    valid_params = inspect.signature(LossClass).parameters
    filtered_params = {k: v for k, v in loss_params.items() if k in valid_params}

    # Reset the loss_type parameter to the Enum name if it exists
    if 'loss_type' in filtered_params:
        filtered_params['loss_type'] = LossCriterion.__members__.get(filtered_params['loss_type']).value
    # Convert samples to a torch tensor if it exists
    if 'samples' in filtered_params:
        filtered_params['samples'] = torch.tensor(filtered_params['samples'], dtype=torch.float32)
    # Convert weights to a torch tensor if it exists
    if 'weights' in filtered_params:
        filtered_params['weights'] = torch.tensor(filtered_params['weights'], dtype=torch.float32)

    # Instnatiate the criterion
    criterion = LossClass(**filtered_params)
    
    return criterion

class CELoss(torch.nn.Module):
    """
    Wrapper class for vanilla cross entropy loss.
    """
    def __init__(
            self, 
            ignore_index=-1, 
            label_smoothing: float=0.0, 
            weights: Optional[torch.tensor]=None, 
            reduction: str='mean', 
            use_weights: bool=True
        ):
        """Instantiate a CELoss object.

        Parameters:
        -----------
            ignore_index : int, optional
                Index to ignore in the target, by default -1
            label_smoothing : float, optional
                A float in the range [0.0, 1.0]. Specifies the amount of smoothing to apply to the labels, by default 0.0
            weights : torch.tensor, optional
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the weight for that class, by default None
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
            use_weights : bool, optional
                Whether to use the provided weights or not. Defaults to True.
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing
        self.weights = weights
        
        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction

        self.use_weights = use_weights
    
    def _mask_targets(self, targets: torch.Tensor, mask: torch.BoolTensor, ignore_index: int=-1) -> torch.tensor:
        """
        Helper function to create a new target with the ignore_index at the masked locations.

        Parameters:
        -----------
            targets : torch.Tensor
                A torch.tensor of shape (N, H, W) and dtype int.
            mask : torch.BoolTensor
                A boolean torch tensor of shape (N, H, W).
            ignore_index (int, optional): Index value to used for the masked values. Defaults to -1.

        Returns:
        --------
            adj_targets : torch.tensor
                A tensor with the masked target values replaced by the ignore index.
        """

        # Create a new target tensor with ignore_index at the masked locations
        adj_targets = torch.where(mask, targets, torch.full_like(targets, ignore_index)).to(targets.device)

        return adj_targets

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of CELoss.

        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
        --------
            loss : torch.tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        # Apply mask if provided
        if mask is not None:
            targets = self._mask_targets(targets, mask, self.ignore_index)
        # Compute the loss
        if self.use_weights:
            loss = F.cross_entropy(
                input=preds, 
                target=targets, 
                ignore_index=self.ignore_index, 
                label_smoothing=self.label_smoothing, 
                reduction=self.reduction, 
                weight=self.weights
            )
        else:
            loss = F.cross_entropy(
                input=preds, 
                target=targets, 
                ignore_index=self.ignore_index, 
                label_smoothing=self.label_smoothing, 
                reduction=self.reduction
            )

        return loss

class FocalLoss(torch.nn.Module):
    """
    Implementation of Focal Loss
    https://arxiv.org/abs/1708.02002
    """
    def __init__(
            self, 
            weights: Union[float, torch.tensor]=1, 
            gamma: float=2, 
            reduction: str='mean'
        ):
        """Instantiate a FocalLoss object.

        Parameters:
        -----------
            weights : float or torch.tensor, optional
                Weights of shape (C,) where C is the number of classes, or a single float for uniform weighting. Defaults to 1.
            gamma : float, optional
                Focusing parameter for modulating factor (1-p). Defaults to 2.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum' , 'none']. Defaults to 'mean'.
            weights (Union[float, torch.tensor], optional): _description_. Defaults to 1.
            gamma (float, optional): _description_. Defaults to 2.
            reduction (str, optional): _description_. Defaults to 'mean'.
        """
        super().__init__()
        self.weights = weights
        self.gamma = gamma
        
        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction
    
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """Forward method of FocalLoss.

        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
        --------
            loss : torch.tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        if isinstance(self.weights, torch.Tensor) and len(self.weights) != preds.shape[1]:
            raise ValueError(f"Length of weights should be 1 or the number of classes in preds, {preds.shape[1]}. Please set new weights.")
        
        # Calculate log_pt and pt from the raw logits
        log_pt = F.log_softmax(preds, dim=1)
        log_pt = torch.gather(log_pt, 1, targets.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        # Set weights weights
        if isinstance(self.weights, torch.Tensor):
            weights = self.weights[targets]
        else:
            weights = self.weights
        # Calculate focal loss
        loss = -weights * ((1 - pt) ** self.gamma) * log_pt

        # Mask the loss if mask is provided
        if mask is not None:
            loss = loss * mask

        # Apply reductions
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class CBLoss(torch.nn.Module):
    """
    Implementation of Class Balanced losses
    https://arxiv.org/pdf/1901.05555
    """
    def __init__(
            self, 
            samples: torch.tensor, 
            loss_type: str, 
            reduction: str='mean', 
            gamma: Optional[float]=2.0
        ):
        """Instantiate a CBLoss object.

        Parameters:
        -----------
            samples : torch.tensor
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum']. Defaults to 'mean'.
            gamma : float, optional
                The gamma value to use for Focal Loss. Only used if loss_type is 'FocalLoss'. Defaults to 2.0.

        Raises:
        -------
            ValueError: If loss_type is not one of ['CELoss', 'FocalLoss'].
        """
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.beta = (self.N - 1) / self.N
        self.C = len(self.samples)
        self.eps = 1e-7

        # Set the loss function with the proper weights
        self._set_effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(weights=self.weights, reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(weights=self.weights, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _set_effective_samples(self):
        """Helper function to calculate the effective samples and weights.
        
        Effective samples E_n = (1 - beta^n) / (1 - beta)
        Weights weights_n = C / (E_n * sum(1/E_n)) such that sum(weights_n) = C
        """
        # Calculate effective samples
        E = (1 - torch.pow(self.beta, self.samples)).double() / (1 - self.beta + self.eps).double()

        # Invert to get weights weights and normalize to sum to C
        self.weights = ((1/E) / (1/E).sum() * self.C).float()

     
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """Forward method of CBLoss.

        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model.
            targets : torch.tensor
                The ground truth targets.
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
        --------
            loss : torch.tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        loss = self.loss_fn(preds=preds, targets=targets, mask=mask)
        
        return loss
  
class ACBLoss(torch.nn.Module):
    """
    Implement Adaptive Class Balanced Loss from Xu et al 2022.
    https://ieeexplore.ieee.org/document/10137858
    """
    def __init__(
            self, 
            samples: torch.tensor, 
            loss_type: str, 
            reduction: str='mean', 
            gamma: Optional[float]=2.0
        ):
        """Instantiate an ACBLoss object.

        Parameters:
        -----------
            samples : torch.tensor
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
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
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.N_max = torch.max(self.samples)
        self.C = len(self.samples)
        self.eps = 1e-7

        self._effective_samples()
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(weights=self.weights.float(), reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(weights=self.weights.float(), gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _effective_samples(self):
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
        # Invert to get weights weights and normalize
        weights = 1/E * self.C / (1/E).sum()

        self.E = E
        self.weights = weights

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of ACBLoss.

        Args:
            preds (torch.tensor): The raw logits from the model.
            targets (torch.tensor): The ground truth targets.
            mask (Optional[torch.BoolTensor], optional): A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
            torch.tensor: A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        
        loss = self.loss_fn(preds=preds, targets=targets, mask=mask)
        
        return loss

class RecallLoss(torch.nn.Module):
    """
    Implementation of Recall Loss with dynamic weighting
    https://arxiv.org/pdf/2106.14917
    """
    def __init__(
            self, 
            samples: torch.tensor, 
            loss_type: str, 
            reduction: str='mean', 
            gamma: Optional[float]=2.0, 
            eps: float=0.0000001
        ):
        """Instantiate a RecallLoss object.

        Parameters:
        -----------
            samples : torch.tensor 
                A 1D tensor of shape (C,) where C is the number of classes. Each value is the number of samples for that class in the training set.
            loss_type : str
                The type of loss to use. Must be one of ['CELoss', 'FocalLoss'].
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum']. Defaults to 'mean'.
            gamma : float, optional
                The gamma value to use for Focal Loss. Only used if loss_type is 'FocalLoss'. Defaults to 2.0.
            eps : float, optional
                A small value to avoid division by zero. Defaults to 0.0000001.
        """
        super().__init__()
        self.samples = samples
        self.loss_type = loss_type
        self.reduction = reduction
        self.gamma = gamma
        self.N = self.samples.sum()
        self.C = len(self.samples)
        self.eps = eps

        # Set the loss function with the proper weights
        if self.loss_type == 'CELoss':
            self.loss_fn = CELoss(weights=None, reduction=self.reduction)
        elif self.loss_type == 'FocalLoss':
            self.loss_fn = FocalLoss(weights=None, gamma=self.gamma, reduction=self.reduction)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}. Must be one of ['CELoss', 'FocalLoss']")

    def _calculate_weights(self, preds: torch.tensor, targets: torch.tensor):
        """Helper function to calculate the recall weights.
        
        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.tensor
                The ground truth targets of shape (N, H, W).
        """
        # Get probs from logits and calculate one-hot tensors
        probs = F.softmax(preds, dim=1)
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
        self.weights = (weights / weights.sum() * self.C).float()
        self.loss_fn.weights = self.weights

    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """
        Forward method of RecallLoss.

        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
        --------
            loss : torch.tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """

        # Calculate effective samples
        self._calculate_weights(preds=preds, targets=targets)
        
        # Compute the loss
        loss = self.loss_fn(preds=preds, targets=targets, mask=mask)
        
        return loss

class DiceLoss(torch.nn.Module):
    """
    Implementation of Dice Loss
    https://arxiv.org/abs/1606.04797
    """
    def __init__(
            self, 
            smooth: float = 1.0, 
            reduction: str = 'mean'
        ):
        """Instantiate a DiceLoss object.

        Parameters:
        -----------
            smooth : float, optional
                A smoothing factor to avoid division by zero. Defaults to 1.0.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
        """
        super().__init__()
        self.smooth = smooth

        # Validate reduction
        if reduction not in ['mean', 'sum', 'none']:
            raise ValueError(f"Invalid reduction mode: {reduction}. Must be one of ['mean', 'sum', 'none']")
        self.reduction = reduction
    
    def forward(self, preds: torch.Tensor, targets: torch.Tensor, mask: Optional[torch.BoolTensor] = None, return_stats: bool=False) -> torch.Tensor:
        """Forward method of DiceLoss.

        Parameters:
        -----------
            preds : torch.Tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.Tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.
            return_stats : bool
                A boolean flag whether to return the intersection and union stats
        Returns:
        --------
            loss :  torch.Tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        # 
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            preds = preds * mask
            targets_one_hot = targets_one_hot * mask

        intersection = (preds * targets_one_hot).sum(dim=(2, 3))
        union = preds.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice_score

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        
        if return_stats:
            return loss, intersection, union
        else:
            return loss

class TverskyLoss(torch.nn.Module):
    """
    Implementation of Tversky Loss
    https://arxiv.org/abs/1706.05721
    """
    def __init__(
            self, 
            weights: float = 0.5, 
            beta: float = 0.5, 
            smooth: float = 1.0, 
            reduction: str = 'mean'
        ):
        """Instantiate a TverskyLoss object.

        Parameters:
        -----------
            weights : float, optional
                Weight for false negatives. Defaults to 0.5.
            beta : float, optional
                Weight for false positives. Defaults to 0.5.
            smooth : float, optional
                A smoothing factor to avoid division by zero. Defaults to 1.0.
            reduction : str, optional
                The reduction method to use. Must be one of ['mean', 'sum', 'none']. Defaults to 'mean'.
        """
        super().__init__()
        self.weights = weights
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, preds: torch.tensor, targets: torch.tensor, mask: Optional[torch.BoolTensor] = None) -> torch.tensor:
        """Forward method of TverskyLoss.

        Parameters:
        -----------
            preds : torch.tensor
                The raw logits from the model of shape (N, C, H, W).
            targets : torch.tensor
                The ground truth targets of shape (N, H, W).
            mask : torch.BoolTensor, optional
                A boolean torch tensor of shape (N, H, W) of pixels to exclude. Defaults to None.

        Returns:
        --------
            loss : torch.tensor
                A scalar loss value if reduction is 'mean' or 'sum', else a loss tensor of shape (N, H, W).
        """
        # Get probabilities from logits and convert targets to one-hot
        preds = torch.softmax(preds, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=preds.shape[1]).permute(0, 3, 1, 2).float()
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            preds = preds * mask
            print(preds.shape)
            targets_one_hot = targets_one_hot * mask

        # Calculate true positives, false negatives, and false positives based on the softmax probabilities
        true_pos = (preds * targets_one_hot).sum(dim=(2, 3))
        false_neg = ((1 - preds) * targets_one_hot).sum(dim=(2, 3))
        false_pos = (preds * (1 - targets_one_hot)).sum(dim=(2, 3))

        # Calculate Tversky index and loss
        tversky_index = (true_pos + self.smooth) / (true_pos + self.weights * false_neg + self.beta * false_pos + self.smooth)
        loss = 1 - tversky_index

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss