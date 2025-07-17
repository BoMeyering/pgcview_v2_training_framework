"""
train.py
Main training script for the PGCView V2 semantic segmentation model
BoMeyering 2025
"""

import torch
import os
import argparse
import omegaconf
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss

# Local imports
from src.models import create_smp_model
from src.datasets import LabeledDataset, UnlabeledDataset
from src.flexmatch import class_beta
from src.trainer import SupervisedTrainer, FlexMatchTrainer
from src.transforms import get_train_transforms, get_val_transforms, get_strong_transforms, get_weak_transforms, set_normalization_values
from src.utils.device import set_torch_device
from src.utils.config import TrainSupervisedConfig

# Create a parser
parser = ArgumentParser(
    prog="train.py",
    description="Main training script for the PGCView V2 semantic segmentation model."
)
# Add arguments for config file and then parse CLI args
parser.add_argument('-c', '--config', type=str, help="The path to the training config YAML file.", default='configs/train_config.yaml')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"The path to the configuration file {args.config} was not found.")

# Read in the configuration file and merge with default dict
yaml_conf = OmegaConf.load(args.config)

print(OmegaConf.to_yaml(yaml_conf))

conf = OmegaConf.merge(OmegaConf.structured(TrainSupervisedConfig), yaml_conf)


def main(conf: omegaconf.OmegaConf=conf):
    
    # Set torch device
    set_torch_device(conf)

    # Set data normalization values
    set_normalization_values(conf)

    print(OmegaConf.to_yaml(conf))

    # Create model
    model = create_smp_model(conf=conf).to(conf.device)

    # Augmentation Pipelines
    train_transforms = get_train_transforms(resize=tuple(conf.images.resize))
    val_transforms = get_val_transforms(resize=tuple(conf.images.resize))
    weak_transforms = get_weak_transforms(resize=tuple(conf.images.resize))
    strong_transforms = get_strong_transforms(resize=tuple(conf.images.resize))

    # Create Datasets
    train_ds = LabeledDataset(
        root_dir=conf.directories.train_labeled_dir,
        transforms=train_transforms
    )

    val_ds = LabeledDataset(
        root_dir=conf.directories.val_dir,
        transforms=val_transforms
    )

    # Create DataLoaders
    train_loader = DataLoader(train_ds, conf.batch_size.labeled, shuffle=True)
    val_loader = DataLoader(val_ds, conf.batch_size.labeled, shuffle=True)

    # Optimizer
    optimizer = SGD(lr=0.001, params=model.parameters())

    # Scheduler
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)
    
    # Criterion
    criterion = CrossEntropyLoss()

    supervised_trainer = SupervisedTrainer(
        "my supervised trainer", 
        conf=conf, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion)
    
    supervised_trainer.train()
    


if __name__ == '__main__':
    main()