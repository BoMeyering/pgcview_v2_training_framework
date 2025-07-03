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

# Local imports
from src.models import create_smp_model
from src.datasets import LabeledDataset, UnlabeledDataset
from src.flexmatch import class_beta
from src.trainer import SupervisedTrainer, FlexMatchTrainer
from src.transforms import get_train_transforms, get_val_transforms, get_strong_transforms, get_weak_transforms

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

# Read in the configuration file
conf = OmegaConf.load(args.config)



def main(conf: omegaconf.OmegaConf=conf):
    
    # Set torch device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create model
    model = create_smp_model(conf=conf).to(device)

    # Augmentation Pipelines
    train_transforms = get_train_transforms(resize=tuple(conf.images.resize))
    val_transforms = get_val_transforms(resize=tuple(conf.images.resize))
    weak_transforms = get_weak_transforms(resize=tuple(conf.images.resize))
    strong_transforms = get_strong_transforms(resize=tuple(conf.images.resize))

    # Create Datasets
    train_l_ds = LabeledDataset(
        root_dir=conf.directories.train_labeled_dir,
        transforms=train_transforms
    )

    train_u_ds = UnlabeledDataset(
        root_dir=conf.directories.train_unlabeled_dir,
        weak_transforms=weak_transforms
    )

    


if __name__ == '__main__':
    main()