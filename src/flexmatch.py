"""
src.class_threshold.py
Dynamic beta tau thresholding class functionality
BoMeyering 2025
"""

import torch
import torch.functional as F
from omegaconf import OmegaConf
from typing import Tuple


def class_beta(logits: torch.Tensor, tau: float=0.85, mapping: str='linear', warmup: bool=True) -> torch.Tensor:
    """
    Functional class Tau calculator

    Args:
        logits (torch.Tensor): A torch.Tensor of the raw model logits of shape (N, C, H, W).
        tau (float, optional): A float value in th interval (0, 1] used for confidence thresholding. Defaults to 0.85.
        mapping (str, optional): An optional string indicating the linear or non-linear mapping function to apply. Defaults to 'linear'.
        warmup (bool, optional): Whether or not beta warmup should be performed. Defaults to True.

    Returns:
        torch.Tensor: torch.Tensor: A torch.Tensor of shape (C) with a beta weight in [0, 1] for each class c.
    """

    # Validate arguments
    if type(logits) != torch.Tensor:
        raise ValueError("Argument 'logits' should be type 'torch.Tensor'. Ensure that you passed the correct logits.")
    elif len(logits.shape) != 4:
        raise ValueError("ARgument 'logits' should be a tensor of shape (N, C, H, W). Please ensure that your logits dimensions are correct.")
    elif logits.shape[1] == 0:
        raise ValueError("Argument 'logits' cannot have a class dimension of 0.")
    elif tau <= 0 or tau > 1:
        raise ValueError("Argument 'tau' must be a float value in the interval (0, 1].)")
    elif not isinstance(mapping, str):
        raise ValueError(f"Argument 'mapping' must be a str not a {type(mapping)}.")
    elif mapping not in ['convex', 'concave', 'linear']:
        raise ValueError(f"Argument 'mapping' must be one of 'convex', 'concave' or 'linear'")
    
    try:
        # Softmax and get probs and predictions
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        max_probs, pred_classes = torch.max(probs, dim=1)
        
        # Grab total number of elements in batch logits
        N = max_probs.numel()

        # Get the confidence mask and apply to the flattened class predictions
        conf_mask = max_probs > tau
        conf_flat = conf_mask.view(-1)
        preds_flat = pred_classes.view(-1)
        masked_preds = preds_flat[conf_flat]

        # Count to get sigma_t vector
        sigma_t = torch.bincount(masked_preds, minlength=num_classes)

        # Create normalized beta_t vector
        if warmup:
            beta_t = sigma_t / torch.max(torch.tensor([sigma_t.max(), N - sigma_t.sum()]))
        else:     
            beta_t = sigma_t / sigma_t.max()
        
        # Apply non-linear mapping functions
        if mapping == 'convex':
            beta_t = beta_t / (2 - beta_t)
        elif mapping == 'concave':
            beta_t = torch.log(beta_t + 1) / torch.log(2)

        return beta_t

    except Exception as e:
        print(f"Encountered an error: {e}")

        return torch.tensor([tau]).expand(num_classes)


@torch.no_grad()
def get_pseudo_labels(tau_vector: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """_summary_

    Args:
        tau_vector (torch.Tensor): _description_
        logits (torch.Tensor): _description_

    Raises:
        ValueError: _description_

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: _description_
    """
    
    if len(logits.shape) != 4:
        raise ValueError(f"Argument 'logits' should be of shape (N, C, H, W), but has {len(logits.shape)} instead")
    
    probs = F.softmax(logits, dim=1)
    max_probs, pseudo_labels = torch.max(probs, dim=1)

    # Create an NxHxW tensor of the correct tau per pixel and then create a boolean mask
    tau_expanded = tau_vector[pseudo_labels]
    tau_mask = max_probs.ge(tau_expanded)

    return pseudo_labels, tau_mask

@torch.no_grad()
def mask_targets(targets: torch.Tensor, mask: torch.Tensor, ignore_index: int=-1):
    """_summary_

    Args:
        targets (torch.Tensor): _description_
        mask (torch.Tensor): _description_
        ignore_index (int, optional): _description_. Defaults to -1.

    Returns:
        _type_: _description_
    """

    adj_targets = torch.where(mask, targets, torch.full_like(targets, ignore_index))

    return adj_targets



if __name__ == '__main__':

   for i in range(1, 20):
       X = torch.randn(5, i, 100, 100)

       beta_t = class_beta(logits=X, tau=0.3)

       print(beta_t)
