"""
scripts/create_datasets_from_json.py
Create labeled and unlabeled datasets from the export JSON file
BoMeyering 2025
"""

# Imports
import json
import os
import shutil
import tempfile
import cv2
import urllib
import dotenv
import time

import numpy as np
import labelbox as lb

from pathlib import Path
from tqdm import tqdm
from PIL import Image
from glob import glob
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split

parser = ArgumentParser(description="Create labeled and unlabeled datasets from the export JSON file")
parser.add_argument('--json_path', type=str, help='Path to the export JSON', default='data/pgcviewv2_label_export.json')
parser.add_argument('--recalculate', action='store_true', help='Recalculate and overwrite existing datasets')
args = parser.parse_args()

# Load the .env file
dotenv.load_dotenv()

# Set constants
API_KEY = os.environ.get('API_KEY')
JSON_PATH = args.json_path
RECALCULATE = args.recalculate
PROJECT = "cma74xh22061607ysdyf4gudl"
RENAMED_IMG_DIR = 'data/PGCView_v2/renamed_images'
LABELED_DIR = 'data/processed/train/labeled'
UNLABELED_DIR = 'data/processed/train/unlabeled'
VAL_DIR = 'data/processed/val'
TEST_DIR = 'data/processed/test'
MAP_PATH = 'metadata/target_mapping.json'

# Initialize Labelbox client
client = lb.Client(api_key=API_KEY)
project = client.get_project(PROJECT)

# Read in export JSON
with open(JSON_PATH, 'r') as f:
    export_dict = json.load(f)

# Read in mapping json
with open(MAP_PATH, 'r') as f:
    map_dict = json.load(f)

# Main processing loop
with tempfile.TemporaryDirectory() as temp_dir:
    print(f"Temporary directory created at: {temp_dir}")
    os.makedirs(Path(temp_dir) / 'images', exist_ok=True)
    os.makedirs(Path(temp_dir) / 'targets', exist_ok=True)
    pbar = tqdm(export_dict['data_rows'].items(), colour='blue')

    for key, data_row in pbar:
        pbar.set_description(f"Processing {key}")
        status = data_row['projects'][PROJECT]['project_details']['workflow_status']
        img_path = Path(RENAMED_IMG_DIR) / key
        if not img_path.exists():
            continue
        if status == 'DONE':
            shutil.copy2(img_path, Path(temp_dir) / 'images' / key) # Move image to a temporary directory
            
            # Set the path for the image target
            target_path = Path(temp_dir) / 'targets' / ".".join([key.split('.')[0], "png"])

            mask_shape = (data_row['media_attributes']['height'], data_row['media_attributes']['width'])
            orientation = data_row['media_attributes']['exif_rotation']

            # Create an empty target array
            target = np.zeros(mask_shape)
            
            # Grab all of the annotation objects
            objects = data_row['projects']['cma74xh22061607ysdyf4gudl']['labels'][0]['annotations']['objects']

            for object in objects:
                if object["annotation_kind"] == "ImageSegmentationMask":

                    # Get the class map index
                    class_name = object['name']
                    class_idx = map_dict.get(class_name)['class_idx']

                    mask_url = object['mask']['url']
                    # Make a request for the image
                    try:
                        req = urllib.request.Request(mask_url, headers=client.headers)
                        obj_mask = Image.open(urllib.request.urlopen(req))
                    except urllib.error.HTTPError as e:
                        print(f"Encountered error retrieving mask {e}. Retrying")
                        time.sleep(5)
                        req = urllib.request.Request(mask_url, headers=client.headers)
                        obj_mask = Image.open(urllib.request.urlopen(req))

                    # Convert mask to numpy array
                    obj_mask = np.array(obj_mask)

                    # Overwrite the target mask with the class index
                    idx = np.where(obj_mask == 255)
                    target[idx] = class_idx
            
            # Apply orientation correction
            if orientation == 2:
                target = cv2.flip(target, 1)
            elif orientation == 3:
                target = cv2.rotate(target, cv2.ROTATE_180)
            elif orientation == 4:
                target = cv2.flip(target, 0)
            elif orientation == 5:
                target = cv2.flip(target, 1)
                target = cv2.rotate(target, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif orientation == 6:
                target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 7:
                target = cv2.flip(target, 1)
                target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                target = cv2.rotate(target, cv2.ROTATE_90_COUNTERCLOCKWISE)

            # Write out the target png image
            cv2.imwrite(str(target_path), target.astype(np.uint8))

        else:
            shutil.copy2(img_path, Path(UNLABELED_DIR) / 'images' / key)
    pbar.close()

    # Split labeled data into train, val, and test sets
    img_keys = [img_name.split('.')[0] for img_name in glob('*.jpg', root_dir=Path(temp_dir) / 'images')]
    train_keys, val_keys = train_test_split(img_keys, train_size=2000, random_state=42)

    print("Moving labeled training data")
    for key in tqdm(train_keys, colour='green'):
        tmp_img_path = Path(temp_dir) / 'images' / f"{key}.jpg"
        tmp_target_path = Path(temp_dir) / 'targets' / f"{key}.png"

        # Move image and target to labeled train directory
        shutil.copy2(tmp_img_path, Path(LABELED_DIR) / 'images' / f"{key}.jpg")
        shutil.copy2(tmp_target_path, Path(LABELED_DIR) / 'targets' / f"{key}.png")

    print("Moving labeled validation data")
    for key in tqdm(val_keys, colour='yellow'):
        tmp_img_path = Path(temp_dir) / 'images' / f"{key}.jpg"
        tmp_target_path = Path(temp_dir) / 'targets' / f"{key}.png"

        # Move image and target to labeled train directory
        shutil.copy2(tmp_img_path, Path(VAL_DIR) / 'images' / f"{key}.jpg")
        shutil.copy2(tmp_target_path, Path(VAL_DIR) / 'targets' / f"{key}.png")

print("Labeled dataset creation complete.")

            

