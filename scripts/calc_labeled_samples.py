"""
scripts/calc_labeled_samples.py
Calculate class sample counts
BoMeyering 2025
"""

import os
import json
import sys
import numpy as np
import cv2
import logging
from argparse import ArgumentParser
from tqdm import tqdm
from glob import glob

# Logging setup
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_handler = logging.StreamHandler(sys.stdout,)
stream_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

parser = ArgumentParser(
    prog="calc_labeled_samples.py",
    description="Calculate class sample counts"
)
parser.add_argument('-d', '--data_dir', type=str, help="The path to the directory containing the labeled images.", required=True)
parser.add_argument('-o', '--output_path', type =str, help="The path to save the output JSON file containing class sample counts.", default='metadata/class_sample_counts.json')
args = parser.parse_args()

# Read in mapping file
with open('data/target_mapping.json', 'r') as f:
    mapping = json.load(f)
logger.info(f"Loaded target mapping from data/target_mapping.json: {mapping}")




def main(args=args):
    # Get list of all image files in the directory
    image_files = [file for file in glob(os.path.join(args.data_dir, '*')) if file.lower().endswith('.png')]
    logger.info(f"Found {len(image_files)} image files in {args.data_dir}")

    # Initialize an array of zeros to hold pixel counts for each class (0-11)
    pixel_counts = np.zeros(12, dtype=np.int64)

    # Iterate through each image and count pixel occurrences per class
    pbar = tqdm(image_files, desc="Processing images", colour='green')
    for image_file in pbar:
        # Read the image
        pbar.set_postfix(file=os.path.basename(image_file))
        image = cv2.imread(image_file, cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning(f"Failed to read image {image_file}. Skipping.")
            continue
        
        # Check if the image is single-channel (grayscale)
        if len(image.shape) != 2:
            logger.warning(f"Image {image_file} is not single-channel. Skipping.")
            continue
        
        # Get unique classes and their counts
        unique, counts = np.unique(image, return_counts=True)
        
        pixel_counts[unique] += counts
        
    logger.info(f"Total pixel counts per class: {pixel_counts.tolist()}")

    # Map pixels counts to target classes using the mapping
    mapped_counts = {}
    for k, v in mapping.items():
        idx = v.get('class_idx', None)
        mapped_counts[k] = {'class_idx': idx, 'pixel_count': int(pixel_counts[idx])}
    logger.info(f"Mapped pixel counts: {mapped_counts}")

    # Save the results to a JSON file
    with open(args.output_path, 'w') as f:
        json.dump(mapped_counts, f, indent=4)
    logger.info(f"Saved class sample counts to {args.output_path}")

if __name__ == '__main__':
    main()
