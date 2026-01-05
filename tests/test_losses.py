"""
tests.test_losses.py
src.losses.py unit tests
BoMeyering 2025
"""

import unittest
import random
import torch
import numpy as np
from copy import deepcopy
from omegaconf import OmegaConf
from glob import glob
from wonderwords import RandomWord
from pathlib import Path
from random import randint
from torch import Tensor
from src.losses import get_loss_criterion, CELoss, FocalLoss, CBLoss, ACBLoss, RecallLoss, DiceLoss, TverskyLoss, TvmfDiceLoss
from src.utils.config import LossCriterion, TrainSupervisedConfig


# Set arbitrary number of classes and batch size
N_CLASSES = random.randint(3, 20)
BATCH_SIZE = random.randint(4, 10)
INPUT_SIZE = random.randint(5, 10)

class TestGetLossCriterion(unittest.TestCase):
    """ Test the get_loss_criterion function with OmegaConf configurations """
    def setUp(self):
        """ Set up the test class """
        self.default_conf = OmegaConf.structured(TrainSupervisedConfig)
    
    def test_fallback_loss(self):
        """ Test fallback to Cross Entropy loss when nothing is specified in the conf """

        # Create a null configuration
        conf = OmegaConf.create(
            {}
        )

        # Merge with the default config and get loss function
        conf = OmegaConf.merge(self.default_conf, conf)
        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=CELoss)

    def test_create_celoss(self):
        """ Create CELoss and assert integrity """
        
        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.CELOSS,
                    "reduction": "mean"
                },
                "device": "cpu"
            }
        )
        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=CELoss)

    def test_create_focalloss(self):
        """ Create FocalLoss and assert integrity """
        
        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.FOCALLOSS,
                    "reduction": "mean"
                },
                "device": "cpu"
            }
        )
        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=FocalLoss)

    def test_create_weighted_celoss(self):
        """ Create weighted CELoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.CELOSS,
                    "reduction": "mean",
                    "use_weights": "True",
                    "weights": np.random.randn(N_CLASSES).tolist()
                },
                "device": "cpu"
            }
        )
        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=CELoss)

    def test_create_weighted_focalloss(self):
        """ Create weighted FocalLoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.FOCALLOSS,
                    "reduction": "mean",
                    "use_weights": True,
                    "weights": np.random.randn(N_CLASSES).tolist()
                },
                "device": "cpu"
            }
        )
        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=FocalLoss)

    def test_create_cbceloss(self):
        """ Create Class Balanced CELoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.CBLOSS,
                    "reduction": "mean",
                    "loss_type": "CELOSS",
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=CBLoss)

    def test_create_cbfocalloss(self):
        """ Create Class Balanced FocalLoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.CBLOSS,
                    "reduction": "mean",
                    "loss_type": "FOCALLOSS",
                    "gamma": 2.0,
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=CBLoss)

    def test_create_acbceloss(self):
        """ Create Adaptive Class Balanced CELoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.ACBLOSS,
                    "reduction": "mean",
                    "loss_type": "CELOSS",
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=ACBLoss)

    def test_create_acbfocalloss(self):
        """ Create Adaptive Class Balanced FocalLoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.ACBLOSS,
                    "reduction": "mean",
                    "loss_type": "FOCALLOSS",
                    "gamma": 2.0,
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=ACBLoss)

    def test_create_recallceloss(self):
        """ Create Recall Weighted CELoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.RECALLLOSS,
                    "reduction": "mean",
                    "loss_type": "CELOSS",
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=RecallLoss)

    def test_create_recallfocalloss(self):
        """ Create Recall Weighted FocalLoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.RECALLLOSS,
                    "reduction": "mean",
                    "loss_type": "FOCALLOSS",
                    "gamma": 2.0,
                    "samples": np.random.randint(100, 200, (N_CLASSES)).tolist()
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=RecallLoss)

    def test_create_diceloss(self):
        """ Create Dice Loss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.DICELOSS,
                    "reduction": "mean",
                    "smooth": 1.0
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=DiceLoss)
    
    def test_create_focaldiceloss(self):
        """ Create focal variant DiceLoss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.DICELOSS,
                    "reduction": "mean",
                    "smooth": 1.0,
                    "gamma": 2.0
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=DiceLoss)

    def test_create_tverskyloss(self):
        """ Create Tversky Loss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.TVERSKYLOSS,
                    "reduction": "mean",
                    "beta": 0.6,
                    "weights": 0.4,
                    "smooth": 1.0
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=TverskyLoss)
    
    def test_create_tvmfdiceloss(self):
        """ Create t-vMF Dice Loss and assert integrity """

        conf = OmegaConf.create(
            {
                "loss": {
                    "name": LossCriterion.TVMFDICELOSS,
                    "reduction": "mean",
                    "kappa": 32,
                    "smooth": 1.0
                },
                "device": "cpu"
            }
        )

        loss_fn = get_loss_criterion(conf=conf)

        self.assertIsInstance(obj=loss_fn, cls=TvmfDiceLoss)

class TestLossOverfit(unittest.TestCase):
    """ Overfit each loss function and test loss reduction """
    def setUp(self):
        """ Set up the test class """
        
        self.weights = torch.tensor(np.random.randn(N_CLASSES), dtype=torch.float32)
        self.samples = torch.tensor(np.random.randint(100, 200, (N_CLASSES)), dtype=torch.float32)

        # Instantiate all of the loss functions
        self.loss_funs = {
            "ce": CELoss(),
            "focal": FocalLoss(),
            "weighted_ce": CELoss(weights=self.weights),
            "weighted_focal": FocalLoss(weights=self.weights),
            "cb_ce": CBLoss(samples=self.samples, loss_type="CELoss"),
            "cb_focal": CBLoss(samples=self.samples, loss_type="FocalLoss"),
            "acb_ce": ACBLoss(samples=self.samples, loss_type="CELoss"),
            "acb_focal": ACBLoss(samples=self.samples, loss_type="FocalLoss"),
            "recall_ce": RecallLoss(samples=self.samples, loss_type="CELoss"),
            "recall_focal": RecallLoss(samples=self.samples, loss_type="FocalLoss"),
            "dice": DiceLoss(),
            "tvmfdice": TvmfDiceLoss(kappa=16)
            # "tversky": TverskyLoss()
        }

    def test_overfit_losses(self):
        """ Overfit all loss functions """
        # Overfit each loss fucntion and test that the final loss decreases

        for name, loss_fn in self.loss_funs.items():
            logits = torch.randn(BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE, requires_grad=True) # Logits of size (BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE)
            targets = torch.randint(0, N_CLASSES, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)) # Targets of size (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
            optim = torch.optim.SGD([logits], lr=0.5)

            # Zero the gradient and calculate loss0
            optim.zero_grad()
            loss0 = loss_fn(logits, targets).item()

            for _ in range(200):
                optim.zero_grad()
                loss = loss_fn(logits, targets)
                loss.backward()
                optim.step()
            self.assertLess(loss.item(), loss0)

class TestLossReductionMode(unittest.TestCase):
    """ Test that each loss function reduction returns the correct shape """

    def setUp(self):
        """ Set up the test class """

        self.logits = torch.randn(BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE, requires_grad=True) # Logits of size (BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE)
        self.targets = torch.randint(0, N_CLASSES, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)) # Targets of size (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
        self.weights = torch.tensor(np.random.randn(N_CLASSES), dtype=torch.float32)
        self.samples = torch.tensor(np.random.randint(100, 200, (N_CLASSES)), dtype=torch.float32)

        # Instantiate all of the loss functions
        self.loss_funs = {
            "ce_mean": CELoss(),
            "ce_sum": CELoss(reduction='sum'),
            "ce_none": CELoss(reduction='none'),
            "focal_mean": FocalLoss(),
            "focal_sum": FocalLoss(reduction='sum'),
            "focal_none": FocalLoss(reduction='none'),
            "weighted_ce_mean": CELoss(weights=self.weights),
            "weighted_ce_sum": CELoss(weights=self.weights, reduction='sum'),
            "weighted_ce_none": CELoss(weights=self.weights, reduction='none'),
            "weighted_focal_mean": FocalLoss(weights=self.weights),
            "weighted_focal_sum": FocalLoss(weights=self.weights, reduction='sum'),
            "weighted_focal_none": FocalLoss(weights=self.weights, reduction='none'),
            "cb_ce_mean": CBLoss(samples=self.samples, loss_type="CELoss"),
            "cb_ce_sum": CBLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "cb_ce_none": CBLoss(samples=self.samples, loss_type="CELoss", reduction='none'),
            "cb_focal_mean": CBLoss(samples=self.samples, loss_type="FocalLoss"),
            "cb_focal_sum": CBLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "cb_focal_none": CBLoss(samples=self.samples, loss_type="FocalLoss", reduction='none'),
            "acb_ce_mean": ACBLoss(samples=self.samples, loss_type="CELoss"),
            "acb_ce_sum": ACBLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "acb_ce_none": ACBLoss(samples=self.samples, loss_type="CELoss", reduction='none'),
            "acb_focal_mean": ACBLoss(samples=self.samples, loss_type="FocalLoss"),
            "acb_focal_sum": ACBLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "acb_focal_none": ACBLoss(samples=self.samples, loss_type="FocalLoss", reduction='none'),
            "recall_ce_mean": RecallLoss(samples=self.samples, loss_type="CELoss"),
            "recall_ce_sum": RecallLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "recall_ce_none": RecallLoss(samples=self.samples, loss_type="CELoss", reduction='none'),
            "recall_focal_mean": RecallLoss(samples=self.samples, loss_type="FocalLoss"),
            "recall_focal_sum": RecallLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "recall_focal_none": RecallLoss(samples=self.samples, loss_type="FocalLoss", reduction='none'),
            "dice_mean": DiceLoss(),
            "dice_sum": DiceLoss(reduction='sum'),
            "dice_none": DiceLoss(reduction='none'),
            "tversky_mean": TverskyLoss(),
            "tversky_sum": TverskyLoss(reduction='sum'),
            "tversky_none": TverskyLoss(reduction='none'),
            "tvmfdice_mean": TvmfDiceLoss(),
            "tvmfdice_sum": TvmfDiceLoss(reduction='sum'),
            "tvmfdice_none": TvmfDiceLoss(reduction='none')
        }

    def test_loss_reductions(self):
        """ Iterate through loss functions and test output shapes """

        for name, loss_fun in self.loss_funs.items():
            loss = loss_fun(self.logits, self.targets)

            if loss_fun.reduction=='mean' or loss_fun.reduction == 'sum':
                self.assertEqual(loss.shape, torch.Size([]))
            elif any([sub in name for sub in ['dice', 'tversky']]) and loss_fun.reduction == 'none':
                self.assertEqual(loss.shape, torch.Size([N_CLASSES]))
            elif loss_fun.reduction == 'none':
                self.assertEqual(loss.shape, self.targets.shape)

class TestExtremeLogits(unittest.TestCase):
    """ Test that the loss is finite with large logits """
    def setUp(self):
        """ Set up the test class """
        
        self.logits = torch.full((BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE), 10000000.0) # Logits of size (BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE)
        self.targets = torch.randint(0, N_CLASSES, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)) # Targets of size (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
        self.weights = torch.tensor(np.random.randn(N_CLASSES), dtype=torch.float32)
        self.samples = torch.tensor(np.random.randint(100, 200, (N_CLASSES)), dtype=torch.float32)

        # Instantiate all of the loss functions
        self.loss_funs = {
            "ce": CELoss(),
            "focal": FocalLoss(),
            "weighted_ce": CELoss(weights=self.weights),
            "weighted_focal": FocalLoss(weights=self.weights),
            "cb_ce": CBLoss(samples=self.samples, loss_type="CELoss"),
            "cb_focal": CBLoss(samples=self.samples, loss_type="FocalLoss"),
            "acb_ce": ACBLoss(samples=self.samples, loss_type="CELoss"),
            "acb_focal": ACBLoss(samples=self.samples, loss_type="FocalLoss"),
            "recall_ce": RecallLoss(samples=self.samples, loss_type="CELoss"),
            "recall_focal": RecallLoss(samples=self.samples, loss_type="FocalLoss"),
            "dice": DiceLoss(),
            "focaldice": DiceLoss(gamma=2.0),
            "tversky": TverskyLoss(),
            "tvmfdice": TvmfDiceLoss()
        }

    def test_extreme_losses(self):
        """ Pass large logits to each loss function """

        for name, loss_fn in self.loss_funs.items():
            loss = loss_fn(self.logits, self.targets)

            self.assertTrue(torch.isfinite(loss))

class TestMaskOperations(unittest.TestCase):
    """ Test that masking operations work according to the design """
    def setUp(self):
        """ Set up the test class """
        
        self.logits = torch.randn(BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE) # Logits of size (BATCH_SIZE, N_CLASSES, INPUT_SIZE, INPUT_SIZE)
        self.targets = torch.randint(0, N_CLASSES, (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)) # Targets of size (BATCH_SIZE, INPUT_SIZE, INPUT_SIZE)
        self.optim = torch.optim.SGD([self.logits], lr=0.5)
        self.weights = torch.tensor(np.random.randn(N_CLASSES), dtype=torch.float32)
        self.samples = torch.tensor(np.random.randint(100, 200, (N_CLASSES)), dtype=torch.float32)

        # Instantiate all of the loss functions
        self.loss_funs = {
            "ce": CELoss(reduction='sum'),
            "focal": FocalLoss(reduction='sum'),
            "weighted_ce": CELoss(weights=self.weights, reduction='sum'),
            "weighted_focal": FocalLoss(weights=self.weights, reduction='sum'),
            "cb_ce": CBLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "cb_focal": CBLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "acb_ce": ACBLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "acb_focal": ACBLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "recall_ce": RecallLoss(samples=self.samples, loss_type="CELoss", reduction='sum'),
            "recall_focal": RecallLoss(samples=self.samples, loss_type="FocalLoss", reduction='sum'),
            "dice": DiceLoss(reduction='sum'),
            "tversky": TverskyLoss(reduction='sum'),
            "tvmfdice": TvmfDiceLoss(reduction='sum')
        }

    def test_masking_operations(self):
        """ Iterate through the loss functions and ensure that masking functions properly """

        for name, loss_fn in self.loss_funs.items():
            mask = torch.randint_like(self.targets, 0, 2, dtype=torch.bool)
            anti_mask = torch.logical_not(mask)

            # Create a copy with no reduction
            loss_fn_noredux = deepcopy(loss_fn)
            loss_fn_noredux.reduction='none'

            if name == 'dice':
                loss_0, int_0, union_0 = loss_fn(self.logits, self.targets, return_stats=True)
                loss_1, int_1, union_1 = loss_fn(self.logits, self.targets, mask, return_stats=True)
                loss_2, int_2, union_2 = loss_fn(self.logits, self.targets, anti_mask, return_stats=True)
                
                # Assert that the masked and anit-masked losses equal the unmasked lossS
                self.assertAlmostEqual(int_0.sum().item(), (int_1 + int_2).sum().item(), places=3)
                self.assertAlmostEqual(union_0.sum().item(), (union_1 + union_2).sum().item(), places=3)
            elif name == 'tversky':
                pass
            elif name == 'tvmfdice':
                pass
            else:
                # Create three sets of losses with and without masks
                loss_0 = loss_fn(self.logits, self.targets)
                loss_1 = loss_fn(self.logits, self.targets, mask)
                loss_2 = loss_fn(self.logits, self.targets, anti_mask)

                # Assert that the loss sum from the masked and antimask are close
                self.assertAlmostEqual(loss_0.item(), (loss_1 + loss_2).item(), places=3)