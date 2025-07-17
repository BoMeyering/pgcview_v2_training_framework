"""
src.transforms.py
Image Augmentations and Transformations
BoMeyering 2025
"""
import albumentations
import logging
import os
import json
import omegaconf
import albumentations as A
from omegaconf import OmegaConf
from albumentations.pytorch import ToTensorV2
from typing import Tuple, Iterable, Union

from src.utils.config import Norm


def get_train_transforms(
        resize: Tuple=(1024, 1024), 
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Return a training transformation for images and targets with a standard set of augmentations
    Intended for fully supervised training.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (1024, 1024).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.GaussianBlur(),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_val_transforms(
        resize: Tuple=(1024, 1024), 
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Return a transform function for validation transforms, i.e. just resize and normalize.
    Intended for fully supervised training.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (1024, 1024).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask'})

    return transforms

def get_strong_transforms(
        resize: Tuple=(1024, 1024), 
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Return a strong training transformation for the unlabeled training images and targets.
    Intended for semi-supervised training.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (1024, 1024).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.SafeRotate(),
        A.HorizontalFlip(),
        A.OneOf(
            [
                A.GlassBlur(p=0.6, sigma=0.3, max_delta=2, iterations=1),
                A.ChannelShuffle(p=0.6),
                A.RGBShift(p=0.6),
            ],
            p=0.8
        ),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ], additional_targets={'target': 'mask', 'conf_mask': 'mask'})

    return transforms

def get_weak_transforms(
        resize: Tuple=(1024, 1024), 
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Return a weak training transformation for the unlabeled training images and targets.
    Intended for semi-supervised training.

    Args:
        resize (Tuple, optional): Tuple of new output height and width. Defaults to (1024, 1024).
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets. 
    """
    transforms = A.Compose([
        A.Resize(*resize, p=1),
        A.Affine(),
        A.HorizontalFlip(),
        A.Normalize(mean=means, std=std),
        ToTensorV2()
    ])

    return transforms

def get_tensor_transforms(
        resize: Union[Tuple, None] = (1024, 1024), 
        normalize: bool=True, 
        means: Iterable=(0.485, 0.456, 0.406), 
        std: Iterable=(0.229, 0.224, 0.225)
    ) -> albumentations.Compose:
    """
    Return a tensor transformation used for testing purposes.
    This will resize and normalize the image if needed. 

    Args:
        resize (Union[Tuple, None], optional): Tuple of new output height and width. Defaults to (1024, 1024).
        normalize (bool, optional): Normalize the image or not. Defaults to True.
        means (Iterable, optional): Tuple of the the RGB channel means to normalize by. Defaults to ImageNet means.
        std (Iterable, optional): Tuple of the RGB channel standard deviations to normalize by. Defaults to ImageNet std.

    Returns:
        albumentations.Compose: A Compose function to use in the datasets
    """
    resize_p = 1 if resize else 0
    norm_p = 1 if normalize else 0

    transforms = A.Compose([
        A.Resize(*resize, p=resize_p),
        A.Normalize(mean=means, std=std, p=norm_p),
        ToTensorV2(p=1.0)
    ], additional_targets={'target': 'mask'})
 
    return transforms

def set_normalization_values(conf: OmegaConf):
    """_summary_

    Args:
        conf (OmegaConf): _description_
    """

    logger = logging.getLogger()
    if not isinstance(conf, omegaconf.dictconfig.DictConfig):
        raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
    
    if 'metadata' in conf:
        if (conf.metadata is not None) and ('norm_path' in conf.metadata):
            if os.path.exists(conf.metadata.norm_path):
                with open(conf.metadata.norm_path, 'r') as f:
                    norm_dict = json.load(f)
                    conf.metadata.norm = Norm(means=norm_dict['means'], std=norm_dict['std'])
            else:
                raise ValueError(f"Path to normalization values ({conf.metadata.norm_path}) does not exist")
        else:
            raise KeyError(f"Key 'norm_path' not found in 'conf.metadata'. Please set 'conf.metadata.norm_path' to the normalization value JSON path.")
    else:
        raise KeyError(f"Key 'metadata.norm_path' not found in 'conf'. Please set 'conf.metadata.norm_path' to the normalization value JSON path.")


def set_torch_device(conf: OmegaConf) -> OmegaConf:
    """_summary_

    Args:
        conf (OmegaConf): _description_

    Returns:
        OmegaConf: _description_
    """

    logger = logging.getLogger()
    if not isinstance(conf, omegaconf.dictconfig.DictConfig):
        raise ValueError(f"Argument 'conf' should be of type 'omegaconf.dictconfig.DictConfig'.")
    
    if 'device' in conf:
        if conf.device == 'cuda':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device != conf.device:
                logger.info("CUDA Device is not available at this time. Falling back to CPU computation.")
            conf.device = device
        elif conf.device not in ['cpu', 'cuda']:
            print("Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda']. Falling back to 'cpu' computation.")
            logger.info("Incorrect value set for 'conf.device'. Must be one of ['cpu', 'cuda']. Falling back to 'cpu' computation.")
            conf.device = 'cpu'
    else:
        logger.info("No value set for 'conf.training.device'. Setting to 'cpu'. If 'cuda' devices are available, please explicitly pass 'device: cuda' in the configuration YAML.")
        conf.device = 'cpu'














