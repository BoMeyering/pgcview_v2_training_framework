"""
src.datasets.py
Dataset Classes
BoMeyering 2025
"""
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))


import os
import torch
import re
import cv2
import albumentations
from torch.utils.data import Dataset
from typing import Union, Any, Tuple
from pathlib import Path
from glob import glob

from src.transforms import get_train_transforms, get_weak_transforms, get_strong_transforms, get_tensor_transforms




class StatDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all images in a single directory for image stat collection
    """

    def __init__(self, root_dir: Union[str, Path]):
        """
        Initialize the class

        Args:
            dir_path (Union[str, Path]): Path to the image directory
        """
        # Set the directories and transforms
        self.img_dir = Path(root_dir)
        self.img_keys = sorted(
            [
                img for img in glob("*", root_dir=self.img_dir) if img.endswith(("png", "jpg"))
            ]
        )
        self.transforms = get_tensor_transforms(normalize=False)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, str]:
        """
        Grab one image from the directory and return as tuple of torch.Tensor image and the image name

        Args:
            index (int): Integer index for the item

        Returns:
            Tuple[torch.Tensor, str]: Tensor image and image name
        """

        # Get the image name
        img_key = self.img_keys[index]

        # Load the image, convert to RGB format
        try:
            img = cv2.imread(
                str(self.img_dir / img_key), 
                cv2.IMREAD_COLOR
            )
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype("float32") / 255.0 # Normalize image to [0, 1]
            tensor_img = self.transforms(image=img)['image'] # Convert to tensor

            return (tensor_img, img_key)
        
        except Exception as e:
            print(e)
            
            return torch.tensor(1), img_key

    def __len__(self):
        """ Return length of StatDataset """

        return len(self.img_keys)


class TargetDataset(Dataset):
    """
    Barebones dataset implemented to iterate through all target label arrays (images) in a single directory for label enumeration
    """

    def __init__(
        self, 
        root_dir: Union[str, Path]
    ):
        """
        Initialize the class

        Args:
            root_dir (Union[str, Path]): Path to the label array directory
        """
        # Set the directories and transforms
        self.img_dir = Path(root_dir)
        self.img_keys = sorted(
            [
                img for img in glob("*", root_dir=self.img_dir) if img.endswith(("png"))
            ]
        )
        self.transforms = get_tensor_transforms(normalize=False)

    def __getitem__(self, index) -> Tuple[torch.Tensor, str]:
        """
        Grab one image from the directory and return as tuple of torch.Tensor image and the image name

        Args:
            index (int): Integer index for the item

        Returns:
            Tuple[torch.Tensor, str]: Tensor image and image name
        """

        # Get the image name
        img_key = self.img_keys[index]

        # Load the image and transform to tensor
        try:
            img = cv2.imread(
                str(self.img_dir / self.img_keys[index]), 
                cv2.IMREAD_GRAYSCALE
            )
            tensor_img = self.transforms(image=img)['image']

            return tensor_img, img_key
        
        except Exception as e:
            print(e)

            return torch.tensor(1), img_key

    def __len__(self):
        """ Return length of TargetDataset """

        return len(self.img_keys)


class LabeledDataset(Dataset):
    """
    Generates a torch Dataset for fully labeled images
    """
    def __init__(
        self, 
        root_dir: Union[Path, str], 
        transforms: albumentations.Compose = get_train_transforms()
    ):
        """
        Initialize the LabeledDataset. Runs through all QC checks to ensure image and target correspondence.

        Args:
            root_dir (Union[Path, str]): The path (pathlib.Path, or str) to the root image directory
            transforms (A.Compose, optional): Albumentations transformation function. Defaults to get_train_transforms().

        Raises:
            NotADirectoryError: If self.img_dir is not a direcctory or does not exist
            NotADirectoryError: If self.target_dir is not a directory or does not exist
            ValueError: If each image in self.img_dir does not have a corresponding target array
            FileExistsError: If both self.img_dir and self.target_dir are empty
            ValueError: If names in the self.img_keys and self.target_keys do not match
        """
        # Set the directories and transforms
        self.img_dir = Path(root_dir) / "images"
        self.target_dir = Path(root_dir) / "targets"
        self.transforms = transforms

        # Check for path integrity
        if not os.path.isdir(self.img_dir):
            raise NotADirectoryError(
                f"Path to img_dir {self.img_dir} does not exist. Please check path integrity."
            )
        elif not os.path.isdir(self.target_dir):
            raise NotADirectoryError(
                f"Path to target_dir {self.target_dir} does not exist. Please check path integrity."
            )

        # Sort image names and target names
        self.img_keys = sorted(
            [img for img in glob("*", root_dir=self.img_dir) if img.lower().endswith(("png", "jpg", "jpeg"))]
        )
        self.target_keys = sorted(
            [t for t in glob("*", root_dir=self.target_dir) if t.lower().endswith(("png"))]
        )

        # Check if the images and targets correspond to each other
        if len(self.img_keys) != len(self.target_keys):
            raise ValueError(
                f"Image keys and target keys are different lengths. Please ensure that each training image in {self.img_dir} has a corresponding target mask in {self.target_dir}"
            )
        elif len(self.img_keys) == 0 | len(self.target_keys) == 0:
            raise FileExistsError(
                f"Image or target directories are empty. Please ensure that the right directories were passed in the configuration file."
            )

        # Check that the name sets are the same for images and targets
        img_set = set([re.sub(r'\.\w+', '', i) for i in self.img_keys])
        target_set = set([re.sub(r'\.\w+', '', i) for i in self.target_keys])

        # Check for 1:1 correspondence between the images and targets
        img_set_diff = img_set.difference(target_set)
        target_set_diff = target_set.difference(img_set)
        if len(img_set_diff) != 0:
            raise ValueError(
                f"The following images have no corresponding target in {self.target_dir}:\n"\
                f"{img_set_diff}\n\n"\
                f"Please ensure that each image in {self.img_dir} has a corresponding target in {self.target_dir} with the same base name."
            )
        elif len(target_set_diff) != 0:
            raise ValueError(
                f"The following targets have no corresponding image in {self.img_dir}:\n"\
                f"{target_set_diff}\n\n"\
                f"Please ensure that each target in {self.target_dir} has a corresponding image in {self.img_dir} with the same base name."
            )

    def __getitem__(self, index: int) -> Tuple:
        """
        Get one labeled image and associated targets based on the index

        Args:
            index (int): Integer index for the item

        Returns:
            Tuple: Tuple
        """

        # grab data keys
        img_key = self.img_keys[index]
        target_key = self.target_keys[index]

        # Paths for images and target
        img_path = Path(self.img_dir) / img_key
        target_path = Path(self.target_dir) / target_key

        # read in images and targets
        img = cv2.imread(str(img_path))
        target = cv2.imread(str(target_path), cv2.IMREAD_GRAYSCALE)

        # transform images and targets
        transformed = self.transforms(image=img, target=target)

        return transformed["image"], transformed["target"]

    def __len__(self):
        """ Return the length of the LabeledDataset """

        return len(self.img_keys)


class UnlabeledDataset(Dataset):
    """ Generates a dataset for unlabeled images """

    def __init__(
        self,
        root_dir: Union[Path, str],
        weak_transforms: albumentations.Compose=get_weak_transforms()
    ):
        """
        Initialize the UnlabeledDataset

        Args:
            root_dir (Union[Path, str]): Root directory to the images
            weak_transforms (A.Compose, optional): Albumentations function for the weak transforms. Defaults to get_weak_transforms().
            strong_transforms (A.Compose, optional): Albumentations fucntion for the strong transforms. Defaults to get_strong_transforms().

        Raises:
            NotADirectoryError: If self.img_dir is not a directory or doens't exist
        """
        # Set directories and transforms
        self.img_dir = Path(root_dir) / "images"
        self.weak_transforms = weak_transforms

        # Check path integrity
        if not os.path.isdir(self.img_dir):
            raise NotADirectoryError(
                f"Path to img_dir {self.img_dir} does not exist. Please check path integrity."
            )
        # Sort image names
        self.img_keys = sorted(
            [img for img in glob("*", root_dir=self.img_dir) if img.lower().endswith(("jpg", "jpeg", "png"))]
        )

    def __getitem__(self, index: int) -> Any:
        """
        Get one unlabeled image

        Args:
            index (int): Inter index for the item

        Returns:
            Any: Anything
        """
        # Get the image name
        img_key = self.img_keys[index]

        # Paths for images and target
        img_path = Path(self.img_dir) / img_key

        # Read in the unlabeled image
        img = cv2.imread(str(img_path))

        # Run weak transforms on image
        weak_img = self.weak_transforms(image=img)['image']

        return weak_img

    def __len__(self):
        """ Return the length of the unlabeled dataset """
        
        return len(self.img_keys)









if __name__ == '__main__':

    u_root = 'data/raw/labeled'
    u_ds = UnlabeledDataset(root_dir=u_root)

    for i in range(len(u_ds)):
        img= u_ds[i]

        print(img)