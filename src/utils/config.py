"""
src.utils.config.py
Configuration File Validation script
BoMeyering 2025
"""

import omegaconf
import logging
import datetime
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from enum import Enum
from typing import List

class ModelArchitecture(Enum):
    """
    Enumeration of valid segmentation model architectures from segmentation_models_pytorch.
    """
    
    DEEPLABV3 = 'DeepLabV3'
    DEEPLABV3PLUS = 'DeepLabV3Plus'
    FPN = 'FPN'
    LINKNET = 'Linknet'
    MANET = 'MAnet'
    PAN = 'PAN'
    PSPNET = 'PSPNet'
    SEGFORMER = 'Segformer'
    UPERNET = 'UPerNet'
    UNET = 'Unet'
    UNETPLUSPLUS = 'UnetPlusPlus'

class LossCriterion(Enum):
    """
    Enumeration of valid loss functions implemented in src.losses
    """
    CELOSS = 'CELoss'
    FOCALLOSS = 'FocalLoss'
    CBLOSS = 'CBLoss'
    ACBLOSS = 'ACBLoss'
    RECALLLOSS = 'RecallLoss'
    DICELOSS = 'DiceLoss'
    TVERSKYLOSS = 'TverskyLoss'

@dataclass
class Images:
    input_channels: int=3
    resize: List[int]=field(default_factory=lambda: [512, 512])

@dataclass
class Directories:
    train_labeled_dir: str='data/processed/train/labeled'
    train_unlabeled_dir: str='data/processed/train/unlabeled'
    val_dir: str='data/processed/val'
    test_dir: str='data/processed/test'
    output_dir: str='outputs'
    checkpoint_dir: str='model_checkpoints'
    log_dir: str='logs/run_logs'

@dataclass
class Training:
    epochs: int=30
    optim_name: str='SGD'
    lr: float=0.001
    momentum: float=0.99
    beta1: float=0.1
    beta2: float=0.2
    nesterov: bool=True

@dataclass
class Loss:
    name: LossCriterion=LossCriterion.CELOSS
    # samples: List[float]=field(default_factory=list)
    # weights: List[float]=field(default_factory=list)
    reduction: str='mean'
    # loss_type: str

@dataclass
class BatchSize:
    labeled: int=2
    unlabeled: int=2

@dataclass
class FlexMatch:
    tau: float=0.95
    mapping: str="linear"
    warmup: bool=True

@dataclass
class ModelConfig:
    encoder_name: str='resnet18'
    encoder_depth: int=5
    encoder_weights: str='imagenet'
    input_channels: int=3
    classes: int=3

@dataclass
class Model:
    architecture: ModelArchitecture=ModelArchitecture.UNET
    config: ModelConfig=field(default_factory=ModelConfig)
    weight_decay: float=0.9
    filter_bias_and_bn: bool=True


@dataclass
class Norm:
    means: List[float]=field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: List[float]=field(default_factory=lambda: [0.229, 0.224, 0.225])

@dataclass
class Metadata:
    norm_path: str='metadata/dataset_norm.json'
    norm: Norm=field(default_factory=Norm)

@dataclass
class TrainSupervisedConfig:
    model_run: str='model_run'
    images: Images=field(default_factory=Images)
    directories: Directories=field(default_factory=Directories)
    training: Training=field(default_factory=Training)
    loss: Loss=field(default_factory=Loss)
    device: str='cpu'
    batch_size: BatchSize=field(default_factory=BatchSize)
    flexmatch: FlexMatch=field(default_factory=FlexMatch)
    model: Model=field(default_factory=Model)
    metadata: Metadata=field(default_factory=Metadata)
    logging_level: str='INFO'

@dataclass
class TrainSemiSupervisedConfig(TrainSupervisedConfig):
    model_run: str='semi_supervised_model_run'
    images: Images=field(default_factory=Images)
    directories: Directories=field(default_factory=Directories)
    training: Training=field(default_factory=Training)
    device: str='cpu'


def set_run_name(conf: OmegaConf):
    """
    Append timestamp to conf.run_name

    Args:
        conf (OmegaConf): OmegaConf configuration dict
    """
    run_name = "_".join([conf.model_run, datetime.datetime.now().isoformat(timespec='seconds', sep='_').replace(":", ".")])
    conf.model_run = run_name