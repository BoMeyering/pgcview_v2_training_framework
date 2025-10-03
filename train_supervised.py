"""
train_supervised.py
Main training script for the PGCView V2 semantic segmentation model
BoMeyering 2025
"""

import torch
import os
import logging
import argparse
import omegaconf
from argparse import ArgumentParser
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss

# Local imports
from src.models import create_smp_model
from src.datasets import LabeledDataset, UnlabeledDataset
from src.flexmatch import class_beta
from src.trainer import SupervisedTrainer, FlexMatchTrainer
from src.losses import get_loss_criterion
from src.parameters import OptimConfig, EMA
from src.transforms import get_train_transforms, get_val_transforms, get_strong_transforms, get_weak_transforms, set_normalization_values
from src.utils.device import set_torch_device
from src.utils.config import TrainSupervisedConfig, set_run_name
from src.utils.loggers import setup_loggers


# Create a parser for command line arguments
parser = ArgumentParser(
    prog="train_supervised.py",
    description="Main training script for the PGCView V2 semantic segmentation model."
)
# Add arguments for config file and then parse CLI args
parser.add_argument('-c', '--config', type=str, help="The path to the training config YAML file.", default='configs/train_config.yaml')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"The path to the configuration file {args.config} was not found.")


#----------------------------------------#
# Set up configuration objects
#----------------------------------------#
# Read in the configuration file and merge with default dict
yaml_conf = OmegaConf.load(args.config) # Load user supplied config file
default_conf = OmegaConf.structured(TrainSupervisedConfig) # Load the default config structure - to fill in any missing args
conf = OmegaConf.merge(default_conf, yaml_conf) # Any args in yaml_conf will override defaults

# Append timestamp to run name
set_run_name(conf)

# Set up loggers
setup_loggers(conf)
logger = logging.getLogger()

# Set torch device
set_torch_device(conf)

# Set data normalization values
set_normalization_values(conf)

#----------------------------------------#
# Main entry point
#----------------------------------------#
def main(conf: omegaconf.OmegaConf=conf):
    """Main function to run the supervised training script

    Run the main training script for supervised training of the PGCView V2 semantic segmentation model.
    Pulls in all of the configurations from the provided config file and sets up the model, datasets, dataloaders,
    optimizer, scheduler, and criterion. Then initializes the SupervisedTrainer class and starts training.

    Parameters:
    -----------
        conf : omegaconf.OmegaConf, optional
            The OmegaConf configuration dictionary, by default conf
    """

    # Log training
    logger.info("Current Training Configuration")
    logger.info("Training Configuration\n"+OmegaConf.to_yaml(conf))

    # Create model
    model = create_smp_model(conf=conf).to(conf.device)
    logger.info(f"Created model {conf.model.architecture.value} with encoder {conf.model.config.encoder_name}")

    # Augmentation Pipelines
    train_transforms = get_train_transforms(resize=tuple(conf.images.resize))
    val_transforms = get_val_transforms(resize=tuple(conf.images.resize))
    test_transforms = get_val_transforms(resize=tuple(conf.images.resize))

    # Create Datasets
    train_ds = LabeledDataset(
        root_dir=conf.directories.train_labeled_dir,
        transforms=train_transforms
    )

    val_ds = LabeledDataset(
        root_dir=conf.directories.val_dir,
        transforms=val_transforms
    )

    test_ds = LabeledDataset(
        root_dir=conf.directories.test_dir,
        transforms=test_transforms
    )

    # Create DataLoaders
    train_loader = DataLoader(train_ds, conf.batch_size.labeled, shuffle=True)
    val_loader = DataLoader(val_ds, conf.batch_size.labeled, shuffle=True)
    test_loader = DataLoader(test_ds, conf.batch_size.labeled, shuffle=True)

    # Optimizer
    optim_config = OptimConfig(conf=conf, model=model)
    model, optimizer, scheduler = optim_config.process()
    
    # Criterion
    criterion = get_loss_criterion(conf)

    # Initialize EMA if specified
    if conf.optimizer.ema:
        ema = EMA(model, decay=conf.optimizer.ema_decay, verbose=True)
        logger.info(f"Exponential Moving Average (EMA) enabled with decay rate {conf.optimizer.ema_decay}.")
    else:
        ema = None

    # Initialize Trainer
    supervised_trainer = SupervisedTrainer(
        name="my supervised trainer", 
        conf=conf, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        ema=ema)
    
    # Start training
    supervised_trainer.train()
    


if __name__ == '__main__':
    main()