"""
tests.test_dataloaders.py
src.dataloaders.py unit tests
BoMeyering 2025
"""

import unittest
from math import ceil
from omegaconf import OmegaConf
from glob import glob
from wonderwords import RandomWord
from pathlib import Path
from typing import Tuple
from random import randint
from torch import Tensor
from torch.utils.data import DataLoader
from src.dataloaders import InfiniteSampler, DataLoaderBalancer
from src.datasets import LabeledDataset, UnlabeledDataset


class TestDataloaderCreation(unittest.TestCase):

    def setUp(self):
        self.conf = OmegaConf.create(
            {
                "directories": {
                    "train_labeled_dir": "data/toy_dataset/train/labeled",
                    "train_unlabeled_dir": "data/toy_dataset/train/unlabeled"
                }
            }
        )

    def test_create_labeled_dataloader(self):
        """ Create labeled dataloader and assert integrity """
        # Create the labeled dataset
        self.l_ds = LabeledDataset(root_dir=self.conf.directories.train_labeled_dir)

        # Grab all of the image and target files
        img_files = [filename for filename in glob("*", root_dir=self.l_ds.img_dir) if filename.lower().endswith(("jpg", "png", "jpeg"))]

        l_dl = DataLoader(
            self.l_ds,
            batch_size=2,
            drop_last=True,
            shuffle=True
        )

        for batch_idx, batch in enumerate(l_dl):
            images, targets = batch
            self.assertIsInstance(images, Tensor)
            self.assertIsInstance(targets, Tensor)

    
    def test_infinite_sampler(self):
        """ Create labeled dataloader with InfiniteSampler """
        # Create the labeled dataset
        self.l_ds = LabeledDataset(root_dir=self.conf.directories.train_labeled_dir)

        l_dl = DataLoader(
            self.l_ds,
            batch_size=2,
            drop_last=True,
            sampler=InfiniteSampler(dataset=self.l_ds)
        )
        iterloader = iter(l_dl)
        for _ in range(3*len(iterloader)):

            batch = next(iterloader)
            self.assertIsInstance(batch, list)

            images, targets = batch
            self.assertIsInstance(images, Tensor)
            self.assertIsInstance(targets, Tensor)


class TestDataloaderBalancer(unittest.TestCase):
    def setUp(self):
        self.conf = OmegaConf.create(
            {
                "directories": {
                    "train_labeled_dir": "data/toy_dataset/train/labeled",
                    "train_unlabeled_dir": "data/toy_dataset/train/unlabeled"
                }
            }
        )

        self.l_ds = LabeledDataset(root_dir=self.conf.directories.train_labeled_dir)
        self.u_ds = UnlabeledDataset(root_dir=self.conf.directories.train_unlabeled_dir)

    def test_create_dl_balancer(self):
        """ Create and test a DataLoaderBalancer """

        l_img = [filename for filename in glob("*", root_dir=self.l_ds.img_dir) if filename.lower().endswith(("jpg", "jpeg", "png"))]
        u_img = [filename for filename in glob("*", root_dir=self.u_ds.img_dir) if filename.lower().endswith(("jpg", "jpeg", "png"))]

        max_ds_length = max(len(l_img), len(u_img))

        # Create balancer with batch_size=2 and drop_last=True and balance
        datasets = [self.l_ds, self.u_ds]
        dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(2, 2), drop_last=True)
        dataloaders, max_dl_length = dl_balancer.balance_loaders()  

        # Assert the the longest dataloader in the balancer is equal to the length of that dataset divided by the batch size
        self.assertEqual(max_ds_length // 2, max_dl_length)

        # Create balancer with batch_size=3 and drop_last=True and balance
        dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(3, 3), drop_last=True)
        dataloaders, max_dl_length = dl_balancer.balance_loaders()  

        self.assertEqual(max_ds_length // 3, max_dl_length)

        # Create balancer with batch_size=3 and drop_last=False and balance
        dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(3, 3), drop_last=False)
        dataloaders, max_dl_length = dl_balancer.balance_loaders()  

        self.assertEqual(ceil(max_ds_length / 3), max_dl_length)

        # Create balancer with wrong batch sizes
        with self.assertRaises(ValueError):
            dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(3, 3, 3), drop_last=False)

        # Batch size larger than len of dataset
        with self.assertRaises(ValueError):
            dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(1000, 1000), drop_last=False)

        # Negative batch sizes
        with self.assertRaises(ValueError):
            dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(10, -10), drop_last=False)

        # Float batch size
        with self.assertRaises(ValueError):
            dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(10, 4.3), drop_last=False)

        # String batch size
        with self.assertRaises(ValueError):
            dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(10, "4"), drop_last=False)

    def test_dl_balancer_iteration(self):
        """ Generate and Test Iteration for a DataLoaderBalancer """
        LBS = 3
        UBS = 2
        # Create balancer with batch_size=2 and drop_last=True and balance
        datasets = [self.l_ds, self.u_ds]
        dl_balancer = DataLoaderBalancer(*datasets, batch_sizes=(LBS, UBS), drop_last=True)
        dataloaders, max_dl_length = dl_balancer.balance_loaders()
        l_loader, u_loader = dataloaders
        l_loader, u_loader = iter(l_loader), iter(u_loader)

        for batch_idx in range(max_dl_length):
            batches = (next(l_loader), next(u_loader))

            l_batch, u_batch = batches

            self.assertEqual(len(l_batch[0]), len(l_batch[1]))
            self.assertEqual(len(l_batch[0]), LBS)

            self.assertEqual(len(u_batch), UBS)

