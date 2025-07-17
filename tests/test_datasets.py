"""
tests.test_datasets.py
src.datasets.py unit tests
BoMeyering 2025
"""

import unittest
from omegaconf import OmegaConf
from glob import glob
from wonderwords import RandomWord
from pathlib import Path
from random import randint
from torch import Tensor

from src.datasets import LabeledDataset, UnlabeledDataset, StatDataset, TargetDataset


class TestDatasetCreation(unittest.TestCase):

    def setUp(self):
        self.conf = OmegaConf.create(
            {
                "directories": {
                    "train_labeled_dir": "data/toy_dataset/train/labeled",
                    "train_unlabeled_dir": "data/toy_dataset/train/unlabeled",
                    "test_dir": "data/toy_dataset/labeled",
                    "val_dir": "data/toy/dataset/labeled",
                    "stat_dir": "data/toy_dataset/all_images",
                    "target_dir": "data/toy_dataset/all_targets"
                }
            }
        )

    def test_labeled_dataset(self):
        """ Create LabeledDataset and assert integrity """
        # Create the labeled dataset
        l_ds = LabeledDataset(root_dir=self.conf.directories.train_labeled_dir)

        # Assert that self.l_ds is a LabeledDataset
        self.assertIsInstance(l_ds, LabeledDataset)

        # Grab all of the image and target files
        img_files = [filename for filename in glob("*", root_dir=l_ds.img_dir) if filename.lower().endswith(("jpg", "png", "jpeg"))]
        target_files = [filename for filename in glob("*", root_dir=l_ds.target_dir) if filename.lower().endswith("png")]

        # Assert that the dataset length is the same
        self.assertEqual(len(l_ds), len(img_files))
        self.assertEqual(len(l_ds), len(target_files))

    def test_unlabeled_dataset(self):
        """ Create UnlabeledDataset and assert integrity """
        # Create the unlabeled dataset
        u_ds = UnlabeledDataset(root_dir=self.conf.directories.train_unlabeled_dir)

        # Assert that self.u_ds is an UnlabeledDataset
        self.assertIsInstance(u_ds, UnlabeledDataset)
        
        # Grab all of the image files
        img_files = [filename for filename in glob("*", root_dir=u_ds.img_dir) if filename.lower().endswith(("jpg", "png", "jpeg"))]
        
        # Assert that the dataset length is the same
        self.assertEqual(len(u_ds), len(img_files))

    def test_stat_dataset(self):
        """ Create StatDataset and assert integrity """
        # Create the stat dataset
        s_ds = StatDataset(root_dir=self.conf.directories.stat_dir)

        # Assert that self.s_ds is a StatDataset
        self.assertIsInstance(s_ds, StatDataset)
        
        # Grab all of the image files
        img_files = [filename for filename in glob("*", root_dir=s_ds.img_dir) if filename.lower().endswith(("jpg", "jpeg", "png"))]
        
        # Assert that the dataset length is the same
        self.assertEqual(len(s_ds), len(img_files))

    def test_target_dataset(self):
        """ Create TargetDataset and assert integrity """
        # Create the target dataset
        t_ds = TargetDataset(root_dir=self.conf.directories.target_dir)

        # Assert that self.u_ds is a TargetDataset
        self.assertIsInstance(t_ds, TargetDataset)
        
        # Grab all of the target files
        target_files = [filename for filename in glob("*", root_dir=t_ds.img_dir) if filename.lower().endswith("png")]

        # Assert that the dataset length is the same
        self.assertEqual(len(t_ds), len(target_files))


class TestDatasetIteration(unittest.TestCase):
    def setUp(self):
        self.conf = OmegaConf.create(
            {
                "directories": {
                    "train_labeled_dir": "data/toy_dataset/train/labeled",
                    "train_unlabeled_dir": "data/toy_dataset/train/unlabeled",
                    "test_dir": "data/toy_dataset/labeled",
                    "val_dir": "data/toy/dataset/labeled",
                    "stat_dir": "data/toy_dataset/all_images",
                    "target_dir": "data/toy_dataset/all_targets"
                }
            }
        )
        self.l_ds = LabeledDataset(root_dir=self.conf.directories.train_labeled_dir)
        self.u_ds = UnlabeledDataset(root_dir=self.conf.directories.train_unlabeled_dir)
        self.s_ds = StatDataset(root_dir=self.conf.directories.stat_dir)
        self.t_ds = TargetDataset(root_dir=self.conf.directories.target_dir)

    def test_labeled_dataset_iteration(self):
        """ Test LabeledDataset Iteration """

        # Test ordered iteration through the dataset
        for i in range(len(self.l_ds)):
            img, target = self.l_ds[i]
            self.assertIsInstance(img, Tensor)
            self.assertIsInstance(target, Tensor)

        # Test random indices
        for _ in range(50):
            idx = randint(0, 2*len(self.l_ds))
            if idx < len(self.l_ds):
                img, target = self.l_ds[idx]
                self.assertIsInstance(img, Tensor)
                self.assertIsInstance(target, Tensor)
            else:
                with self.assertRaises(IndexError):
                    img, target = self.l_ds[idx]

    def test_unlabeled_dataset_iteration(self):
        """ Test UnlabeledDataset Iteration """

        # Test ordered iteration through the dataset
        for i in range(len(self.u_ds)):
            img = self.u_ds[i]
            self.assertIsInstance(img, Tensor)

        # Test random indices
        for _ in range(50):
            idx = randint(0, 2*len(self.u_ds))
            if idx < len(self.u_ds):
                img = self.u_ds[idx]
                self.assertIsInstance(img, Tensor)
            else:
                with self.assertRaises(IndexError):
                    img = self.u_ds[idx]

    def test_stat_dataset_iteration(self):
        """ Test StatDataset Iteration """

        # Test ordered iteration through the dataset
        for i in range(len(self.s_ds)):
            batch = self.s_ds[i]
            self.assertIsInstance(batch, dict)
            
            img, img_key, is_error, errors = batch['img'], batch['img_key'], batch['is_error'], batch['errors']
            self.assertIsInstance(img, Tensor)
            self.assertIsInstance(img_key, str)
            self.assertIsInstance(is_error, bool)
            self.assertIsInstance(errors, str)

        # Test random indices
        for _ in range(50):
            idx = randint(0, 2*len(self.s_ds))
            if idx < len(self.s_ds):
                batch = self.s_ds[idx]
                self.assertIsInstance(batch, dict)
            
                img, img_key, is_error, errors = batch['img'], batch['img_key'], batch['is_error'], batch['errors']
                self.assertIsInstance(img, Tensor)
                self.assertIsInstance(img_key, str)
                self.assertIsInstance(is_error, bool)
                self.assertIsInstance(errors, str)
            else:
                with self.assertRaises(IndexError):
                    batch = self.s_ds[idx]

    def test_target_dataset_iteration(self):
        """ Test TargetDataset Iteration """

        # Test ordered iteration through the dataset
        for i in range(len(self.t_ds)):
            img, img_key = self.t_ds[i]
            self.assertIsInstance(img, Tensor)
            self.assertIsInstance(img_key, str)

        # Test random indices
        for _ in range(50):
            idx = randint(0, 2*len(self.t_ds))
            if idx < len(self.t_ds):
                img, img_key = self.t_ds[i]
                self.assertIsInstance(img, Tensor)
                self.assertIsInstance(img_key, str)
            else:
                with self.assertRaises(IndexError):
                    img, img_key = self.t_ds[idx]

