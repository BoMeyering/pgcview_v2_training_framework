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
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP

# Local imports
from src.models import create_smp_model
from src.datasets import LabeledDataset, UnlabeledDataset
from src.flexmatch import class_beta
from src.trainer import SupervisedTrainer, FlexMatchTrainer
from src.metrics import MetricLogger, MeterSet, RunningAvgMeter, ValueMeter
from src.losses import get_loss_criterion, read_class_counts
from src.parameters import OptimConfig, EMA
from src.transforms import get_train_transforms, get_val_transforms, get_strong_transforms, get_weak_transforms, set_normalization_values
from src.utils.device import set_torch_device
from src.utils.config import TrainSupervisedConfig, set_run_name
from src.utils.loggers import setup_loggers, rank_log
from src.distributed import set_env_ranks, setup_ddp



# Create a parser for command line arguments
parser = ArgumentParser(
    prog="train_supervised.py",
    description="Main training script for the PGCView V2 semantic segmentation model."
)
# Add arguments for config file and then parse CLI args
parser.add_argument('-c', '--config', type=str, help="The path to the training config YAML file.", default='configs/train_config.yaml')
parser.add_argument('-b', '--backend', type=str, help="The backend to use for torchrun. Defaults to 'gloo'", default='gloo')
args = parser.parse_args()

if not os.path.exists(args.config):
    raise FileNotFoundError(f"The path to the configuration file {args.config} was not found.")

# Set the backend engine for torchrun
backend = args.backend
setup_ddp(backend=backend)

#----------------------------------------#
# Set up configuration objects
#----------------------------------------#
# Read in the configuration file and merge with default dict
yaml_conf = OmegaConf.load(args.config) # Load user supplied config file
default_conf = OmegaConf.structured(TrainSupervisedConfig) # Load the default config structure - to fill in any missing args
conf = OmegaConf.merge(default_conf, yaml_conf) # Any args in yaml_conf will override defaults

# Set the ranks and world_size
set_env_ranks(conf)

# Append timestamp to run name
set_run_name(conf)

# Set up loggers
setup_loggers(conf)
logger = logging.getLogger()

# Set torch device - will set conf.device as 'TYPE:LOCAL_RANK' e.g. 'cuda:0', 'cpu:2' etc
set_torch_device(conf)

# Set data normalization values
set_normalization_values(conf)

# Read class counts from file
samples, inv_weights = read_class_counts(conf.loss.class_sample_count_path)
conf.loss.samples, conf.loss.weights = samples, inv_weights

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
    rank_log(conf.local_rank, logger.info, "Current Training Configurration\n"+OmegaConf.to_yaml(conf))

    # Create and wrap model for DDP
    model = create_smp_model(conf=conf).to(conf.device)
    model = DDP(
        model, 
        device_ids=[conf.local_rank] if 'cuda' in conf.device else None, 
        output_device=conf.local_rank if 'cuda' in conf.device else None, 
        find_unused_parameters=True
    )
    rank_log(conf.local_rank, logger.info, f"Created model {conf.model.architecture.value} with encoder {conf.model.config.encoder_name} and placed on device {conf.device}")
    rank_log(conf.local_rank, logger.info, f"Other processes have the model placed on device 'device.type:local_rank'")

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

    # Create distributed Samplers
    train_sampler = DistributedSampler(
        dataset=train_ds, 
        rank=conf.local_rank, 
        shuffle=True, 
        drop_last=True
    )
    val_sampler = DistributedSampler(
        dataset=val_ds, 
        rank=conf.local_rank, 
        shuffle=False, 
        drop_last=False
    )
    test_sampler = DistributedSampler(
        dataset=test_ds, 
        rank=conf.local_rank, 
        shuffle=False, 
        drop_last=False
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        dataset=train_ds, 
        batch_size=conf.batch_size.labeled, 
        shuffle=False,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        dataset=val_ds, 
        batch_size=conf.batch_size.labeled,
        shuffle=False,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        dataset=test_ds, 
        batch_size=conf.batch_size.labeled, 
        shuffle=False,
        sampler=test_sampler
    )

    # Optimizer
    optim_config = OptimConfig(conf=conf, model=model)
    model, optimizer, scheduler = optim_config.process()
    
    # Criterion
    criterion = get_loss_criterion(conf)

    # Initialize EMA if specified
    if conf.optimizer.ema:
        ema = EMA(model, decay=conf.optimizer.ema_decay, verbose=True)
        rank_log(conf.local_rank, logger.info, f"Exponential Moving Average (EMA) enabled with decay rate {conf.optimizer.ema_decay}.")
    else:
        ema = None

    # Create MeterSet
    meters = MeterSet({
        'train_loss': ValueMeter(),
        'val_loss': ValueMeter(),
        'train_loss_smooth': RunningAvgMeter(window_length=15),
        'val_loss_smooth': RunningAvgMeter(window_length=15)
    })

    # Initialize Trainer
    supervised_trainer = SupervisedTrainer(
        name="my supervised trainer",
        meter_set=meters,
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