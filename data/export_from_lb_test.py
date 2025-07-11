# Export this project.

import urllib
import json
import os
import time
import urllib.error
import numpy as np
from PIL import Image, ImageOps
from pathlib import Path
import labelbox as lb
import cv2
import sys
import shutil



client = lb.Client(api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbGhpNmJiY2swMGt5MDd3YzlncnA1Z2g4Iiwib3JnYW5pemF0aW9uSWQiOiJjbGhpNmJiYzMwMGt4MDd3Y2Q0aG84cnhoIiwiYXBpS2V5SWQiOiJjbWNqazZ0M3EwMzdqMDd4c2I3eGhhOTY0Iiwic2VjcmV0IjoiNjg4YzI2ODI3N2QyZjE2MTcxMTliNTQyMGU2YjRkYjMiLCJpYXQiOjE3NTEzMTU3NjUsImV4cCI6MTc1Njc1ODk2NX0.3YQtosOmslUIKjBmL4SBrDUch8Devjt7gH3gCje1ITE')
params = {}

project = client.get_project('cma74xh22061607ysdyf4gudl')

# Set the export params to include/exclude certain fields. 
export_params= {
  "attachments": False,
  "metadata_fields": True,
  "data_row_details": False,
  "project_details": True,
  "label_details": True,
  "performance_details": False,
  "interpolated_frames": False
}

# Note: Filters follow AND logic, so typically using one filter is sufficient.
filters= {"workflow_status": "Done"}

print("Currently Exporting")
export_task = project.export(params=export_params, filters=filters)
export_task.wait_till_done()

# Stream the export using a callback function
def json_stream_handler(output: lb.BufferedJsonConverterOutput):
  print(output.json)

export_task.get_buffered_stream(stream_type=lb.StreamType.RESULT).start(stream_handler=json_stream_handler)

# Collect all exported data into a list
export_json = [data_row.json for data_row in export_task.get_buffered_stream()]

print("Export Complete!")
with open('data/target_mapping.json', 'r') as f:
    map_dict = json.load(f)

print(map_dict)

RENAMED_IMAGE_DIR = Path('data/PGCView_v2/renamed_images')

for row in export_json:
    image_url = row['data_row']['row_data']
    external_id = (row['data_row']['external_id'])

    print(external_id)

    # Image path to labeled images
    image_path = Path('data/raw/labeled/images') / external_id
    if not os.path.exists(image_path):
        shutil.copy2(src=RENAMED_IMAGE_DIR/external_id, dst=image_path)

    # Target path to labeled targets
    output_path = Path('data/raw/labeled/targets') / ".".join([external_id.split('.')[0], "png"])
    if not os.path.exists(output_path):
        mask_shape = (row['media_attributes']['height'], row['media_attributes']['width'])
        orientation = row['media_attributes']['exif_rotation']

        # Create an empty target array
        target = np.zeros(mask_shape)
        
        # Grab all of the annotation objects
        objects = row['projects']['cma74xh22061607ysdyf4gudl']['labels'][0]['annotations']['objects']

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
                    print(f"Encountered error retrieving mask. Retrying")
                    time.sleep(5)
                    req = urllib.request.Request(mask_url, headers=client.headers)
                    obj_mask = Image.open(urllib.request.urlopen(req))

                # obj_mask = ImageOps.exif_transpose(obj_mask)
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

        # Write out the target image
        cv2.imwrite(output_path, target.astype(np.uint8))

