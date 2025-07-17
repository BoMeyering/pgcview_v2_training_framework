"""
scripts/calc_mean_and_std.py
Script to calculate the dataset image RGB means and standard deviations for image normalization

BoMeyering 2025
"""

# Import modules
import torch
import os
import sys
import json
import argparse
import logging
import datetime
import warnings
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from typing import Tuple
from logging import getLogger

# Append current directory to path so we can find src
script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.dirname(script_directory)
sys.path.append(root_directory)

# Import local code
# from src.utils.config import YamlConfigLoader, ArgsAttributes
from src.datasets import StatDataset
from src.utils.welford import WelfordCalculator

# Set up logger functions and output
logger = logging.getLogger('norm_calc_logger')
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

# File handler
file_handler = logging.FileHandler('logs/norm_calc_log_'+datetime.datetime.now().isoformat(timespec='seconds', sep="_")+'.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Stream handler
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grab config and parse
parser = argparse.ArgumentParser(description="Use this script to calculate the RGB channel means and standard deviations")
parser.add_argument('-c', '--config', help="The path to the yaml configuration file which defines the directories containing the images")
parser.add_argument('-r', '--recalculate', default=False, help="Boolean. Should previously calcualted values be overwritten?")
args = parser.parse_args()

# Set image directory paths to args
if not os.path.exists(args.config):
    raise FileNotFoundError(f"Path to config file {args.config} does not exist. Please specify a different path")
conf = OmegaConf.load(args.config)

# Set device dynamically with args
if 'device' in conf:
    print(True)
    if conf.device == 'cuda':
        if torch.cuda.is_available():
            try:
                device = torch.device('cuda')
            except RuntimeError as e:
                logger.info(f"{e}\nFalling back to CPU computation.")
                device = torch.device('cpu')
        else:
            logger.info("CUDA requested but not available. "
                  "Please check your NVIDIA GPU and driver: http://www.nvidia.com/Download/index.aspx")
            device = torch.device('cpu')
    else:
        device = torch.device(conf.device)
else:
    device = torch.device('cpu')  # fallback if 'device' not specified

def main(conf: OmegaConf) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute channel-wise pixel mean and std over a list of datasets in args

    Args:
        args (argparse.Namespace): Args containing key 'training_dirs' with a list value containing all the image directories used for training

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tensors of channel wise mean and std in RGB format
    """

    # Initialize Welford calculator
    welford = WelfordCalculator(device=device)

    print(conf)
    if 'image_dirs' not in conf:
        raise ValueError(f'{args.config} does not contain the key "image_dirs"')
    # Loop over all the image directories
    for path in conf.image_dirs:
        if not os.path.exists(path):
            # Warn user and continue to next directory
            warnings.warn(f'The directory {path} does not exist. Skipping for now.', Warning)
            continue
        logger.info(f"Calculating mean and std for path {path}")

        # Create a stat dataset
        ds = StatDataset(root_dir=path)
        dl = DataLoader(
            ds, 
            shuffle=False, 
            num_workers=2,
            pin_memory=True,
            batch_size=1
        )

        iter_loader = iter(dl)

        # Main directory loop
        pbar = tqdm(total=len(dl), desc="Overall Progress", unit="image")
        for batch in iter_loader:
            # unpack the batch
            img, img_key, is_error = batch['img'], batch['img_key'][0], batch['is_error'].item()
            pbar.set_description(f"Processing {img_key}")
            if is_error:
                logger.debug(f"Experienced errors when fetching {img_key} at {path}. Errors: {batch['errors']}")
                pbar.update(1)
                continue
                
            if len(img.shape) < 3:
                logger.debug(f"Image {img_key} in dataset {path} is corrupt. Please check the image integrity")
                pbar.update(1)
                continue
            
            img = img.squeeze().to(device)
            welford.update(img)
            pbar.update(1)
        logger.info(f"Finished incorporating data from {path} into mean and std calculations")
        pbar.close()

    mean, std = welford.compute()
     
    return mean, std

if __name__ == '__main__':

    logger.info(f'Calculating the RGB channel means and standard deviations')
    
    # Check if recalculating values
    if args.recalculate:
        # Run main and get means, std
        means, std = main(conf)
    elif os.path.exists(conf.out_path):
        logger.info(f'{conf.out_path} already exists. Use the flag "--recalculate" if you want to overwrite these values')
        sys.exit(0)
    else:
        means, std = main(conf)

    # Build a results dict and export to JSON
    norm_dict = {
        'means': [i.item() for i in means],
        'std': [i.item() for i in std]
    }

    with open(conf.out_path, 'w') as f:
        json.dump(norm_dict, f)
