
import os
from glob import glob
from pathlib import Path
from shutil import copy2

LABELED_ROOT = Path('data/processed/train/labeled/')
UNLABELED_ROOT = Path('data/processed/train/unlabeled')

P_LABELED = Path('data/processed/train/labeled')
P_UNLABELED = Path('data/processed/train/unlabeled')
R_LABELED = Path('data/raw/labeled')
PGC_UNLABELED = Path('data/PGCView_v2/renamed_images')

# Filename lists
rl_images = [file for file in glob("*", root_dir=R_LABELED / "images") if file.lower().endswith(("jpg", "jpeg", "png"))]
rl_targets = [file for file in glob("*", root_dir=R_LABELED / "targets") if file.lower().endswith("png")]

pl_images = [file for file in glob("*", root_dir=P_LABELED / "images") if file.lower().endswith(("jpg", "jpeg", "png"))]
pl_targets = [file for file in glob("*", root_dir=P_LABELED / "targets") if file.lower().endswith("png")]

pgcu_images = [file for file in glob("*", root_dir=PGC_UNLABELED) if file.lower().endswith(("jpg", "jpeg", "png"))]
pu_images = [file for file in glob("*", root_dir=P_UNLABELED/ "images") if file.lower().endswith(("jpg", "jpeg", "png"))]



# Current labeled and unlabled images
labeled_images = [file for file in glob("*", root_dir=LABELED_ROOT / "images") if file.lower().endswith(("jpg", "jpeg", "png"))]
labeled_targets = [file for file in glob("*", root_dir=LABELED_ROOT / "targets") if file.lower().endswith("png")]
unlabeled_images = [file for file in glob("*", root_dir=UNLABELED_ROOT / "images") if file.lower().endswith(("jpg", "jpeg", "png"))]


print(f"There are {len(pgcu_images)} images in total in the PGCView_v2 dataset")
print(f"There are currently {len(pu_images)} images in the processed unlabeled data folder")

print(f"There are currently {len(rl_images)} images and {len(rl_targets)} in the raw labeled folder")
print(f"There are currently {len(pl_images)} images and {len(pl_targets)} in the processed labeled folder")

print(f"There are a total of {len(pu_images) + len(pl_images)} images in total that are in the processed folder")

# Check for any new images
all_processed = pl_images + pu_images
new_pgc_img = list(set(pgcu_images).difference(set(all_processed)))

if len(new_pgc_img) > 0:
    print(f"There are {len(new_pgc_img)} in the PGCView_v2 dataset that will be moved to processed unlabeled")
    for filename in new_pgc_img:
        copy2(src=PGC_UNLABELED / filename, dst=P_UNLABELED / 'images' / filename)


# Check for newly labeled images in raw labeled
newly_labeled_img = list(set(rl_images).difference(set(pl_images)))
newly_labeled_targets = list(set(rl_targets).difference(set(pl_targets)))

print(f"There are {len(newly_labeled_img)} images and {len(newly_labeled_targets)} targets in raw-labeled that should be moved to processed labeled")

if (len(newly_labeled_img) > 0) & (len(newly_labeled_img) != len(newly_labeled_targets)):
    print(f"Please check the labelbox exporter as the number of images and targets in the raw labeled dataset are different")
    SystemExit
elif (len(newly_labeled_img) > 0) & (len(newly_labeled_img) == len(newly_labeled_targets)):
    print("Moving new labeled images and targets to processed labeled directory")
    for filename in newly_labeled_img:
        copy2(src=R_LABELED / 'images' / filename, dst=P_LABELED / 'images' / filename)
    for filename in newly_labeled_targets:
        copy2(src=R_LABELED / 'targets' /filename, dst=P_LABELED / 'targets' / filename)

    print("Done Moving!")

# Finding images to remove from processed unlabeled
pu_to_delete = list(set(pl_images).intersection(set(pu_images)))
if len(pu_to_delete) > 0:
    print(f"There are {len(pu_to_delete)} images that need to be deleted out of the processed unlabeled folder")
    for filename in pu_to_delete:
        os.remove(P_UNLABELED / 'images' / filename)
    print(f"Removed {len(pu_to_delete)} images from processed unlabeled.")
