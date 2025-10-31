"""
data/export_and_move_images.py

This script exports the most current labels from the labelbox project
Loops over them to determine which images have been annotated and which ones haven't
Annotated images will be moved to data/raw/labeled
Unlabeled images will be moved to data/raw/unlabeled

BoMeyering 2025
"""
import dotenv
from tqdm import tqdm
from datetime import datetime, timedelta
import time
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

dotenv.load_dotenv()

API_KEY = os.environ.get('API_KEY')
OUTPUT_JSON_PATH = 'data/pgcviewv2_label_export.json'
PROJECT_KEY = 'cma74xh22061607ysdyf4gudl'

client = lb.Client(api_key=API_KEY)
project = client.get_project(PROJECT_KEY)

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
# filters= {"workflow_status": "Done"}

now = datetime.now().replace(microsecond=0)
EXPORT = True
if os.path.exists(OUTPUT_JSON_PATH):
    with open(OUTPUT_JSON_PATH, 'r') as f:
        data = json.load(f)
        if 'export_time' in data:
            export_time = datetime.strptime(data['export_time'], "%Y-%m-%d %H:%M:%S")
            time_diff = now - export_time
            if time_diff < timedelta(days=0, hours=6):
                EXPORT = False

if EXPORT:
    print(f'Currently exporting data rows from project {PROJECT_KEY}')
    export_task = project.export(params=export_params) # Grab all images in the project
    export_task.wait_till_done()

    export_json = {
        'export_time': str(now),
        'data_rows': {}
    }

    for data_row in tqdm(export_task.get_buffered_stream(), colour='green', desc='Exporting Data Rows'):
        data_row = data_row.json
        external_id = data_row['data_row']['external_id']
        
        export_json['data_rows'][external_id] = data_row

    with open(OUTPUT_JSON_PATH, 'w') as f:
        json.dump(export_json, f)
    print("Export Complete!")
else:
    sys.exit(0)


# export_task = project.export(params=export_params, filters=filters)


# Stream the export using a callback function
# def json_stream_handler(output: lb.BufferedJsonConverterOutput):
#     print(output.json)




# for data_row in export_task.get_buffered_stream():
        


# export_task.get_buffered_stream(stream_type=lb.StreamType.RESULT).start(stream_handler=json_stream_handler)

# Collect all exported data into a list
# export_json = [data_row.json for data_row in export_task.get_buffered_stream()]

# print("Export Complete!")
# with open('data/target_mapping.json', 'r') as f:
#         map_dict = json.load(f)

# print(map_dict)

# for i, row in enumerate(export_json):
#     image_url = row['data_row']['row_data']
#     external_id = (row['data_row']['external_id'])

# with open('pgcviewv2_export.json', 'w') as f:
#    json.dump(export_json, f)

    # # Image path to labeled images
    # image_path = Path('data/raw/labeled/images') / external_id
    # if not os.path.exists(image_path):
    #     shutil.copy2(src=RENAMED_IMAGE_DIR/external_id, dst=image_path)

    # # Target path to labeled targets
    # output_path = Path('data/raw/labeled/targets') / ".".join([external_id.split('.')[0], "png"])
    # if not os.path.exists(output_path):
    #     mask_shape = (row['media_attributes']['height'], row['media_attributes']['width'])
    #     orientation = row['media_attributes']['exif_rotation']

    #     # Create an empty target array
    #     target = np.zeros(mask_shape)
        
    #     # Grab all of the annotation objects
    #     objects = row['projects']['cma74xh22061607ysdyf4gudl']['labels'][0]['annotations']['objects']

    #     for object in objects:
    #         if object["annotation_kind"] == "ImageSegmentationMask":

    #             # Get the class map index
    #             class_name = object['name']
    #             class_idx = map_dict.get(class_name)['class_idx']

    #             mask_url = object['mask']['url']
    #             # Make a request for the image
    #             try:
    #                 req = urllib.request.Request(mask_url, headers=client.headers)
    #                 obj_mask = Image.open(urllib.request.urlopen(req))
    #             except urllib.error.HTTPError as e:
    #                 print(f"Encountered error retrieving mask. Retrying")
    #                 time.sleep(5)
    #                 req = urllib.request.Request(mask_url, headers=client.headers)
    #                 obj_mask = Image.open(urllib.request.urlopen(req))

    #             # obj_mask = ImageOps.exif_transpose(obj_mask)
    #             obj_mask = np.array(obj_mask)

    #             # Overwrite the target mask with the class index
    #             idx = np.where(obj_mask == 255)
    #             target[idx] = class_idx
        
    #     # Apply orientation correction
    #     if orientation == 2:
    #         target = cv2.flip(target, 1)
    #     elif orientation == 3:
    #         target = cv2.rotate(target, cv2.ROTATE_180)
    #     elif orientation == 4:
    #         target = cv2.flip(target, 0)
    #     elif orientation == 5:
    #         target = cv2.flip(target, 1)
    #         target = cv2.rotate(target, cv2.ROTATE_90_COUNTERCLOCKWISE)
    #     elif orientation == 6:
    #         target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
    #     elif orientation == 7:
    #         target = cv2.flip(target, 1)
    #         target = cv2.rotate(target, cv2.ROTATE_90_CLOCKWISE)
    #     elif orientation == 8:
    #         target = cv2.rotate(target, cv2.ROTATE_90_COUNTERCLOCKWISE)

    #     # Write out the target image
    #     cv2.imwrite(output_path, target.astype(np.uint8))

