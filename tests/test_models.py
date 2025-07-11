"""
tests.test_models.py
models.py unit tests
BoMeyering 2025
"""

import unittest
import segmentation_models_pytorch
from omegaconf import OmegaConf
from wonderwords import RandomWord

from src.models import create_smp_model


class TestModelCreation(unittest.TestCase):

    def setUp(self):
        self.conf = OmegaConf.create(
            {
                "model": {
                    "architecture": "FPN",
                    "config": {
                        "encoder_name": "resnet18",
                        "encoder_depth": 5,
                        "encoder_weights": None,
                        "in_channels": 3,
                        "classes": 4
                    }
                }
            }
        )

    def test_unet(self):
        """ Test Create Unet """
        self.conf.model.architecture = 'Unet'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.unet.model.Unet
        )

    def test_unetplusplus(self):
        """ Test Create UnetPlusPlus"""
        self.conf.model.architecture = 'UnetPlusPlus'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.unetplusplus.model.UnetPlusPlus
        )

    def test_manet(self):
        """ Test Create MAnet """
        self.conf.model.architecture = 'MAnet'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.manet.model.MAnet
        )

    def test_linknet(self):
        """ Test Create Linknet """
        self.conf.model.architecture = 'Linknet'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.linknet.model.Linknet
        )

    def test_fpn(self):
        """ Test Create FPN """
        self.conf.model.architecture = 'FPN'
        model = create_smp_model(self.conf)
        
        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.fpn.model.FPN
        )

    def test_pspnet(self):
        """ Test Create PSPNet """
        self.conf.model.architecture = 'PSPNet'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.pspnet.model.PSPNet
        )

    def test_pan(self):
        """ Test Create PAN"""
        self.conf.model.architecture = 'PAN'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.pan.model.PAN
        )

    def test_deeplabv3(self):
        """ Test Create DeepLabV3 """
        self.conf.model.architecture = 'DeepLabV3'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.deeplabv3.model.DeepLabV3
        )

    def test_upernet(self):
        """ Test Create UPerNet """
        self.conf.model.architecture = 'UPerNet'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.upernet.model.UPerNet
        )

    def test_segformer(self):
        """ Test Create Segformer """
        self.conf.model.architecture = 'Segformer'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.segformer.model.Segformer
        )

    def test_deeplabv3plus(self):
        """ Test Create DeepLabV3Plus """
        self.conf.model.architecture = 'DeepLabV3Plus'
        model = create_smp_model(self.conf)

        self.assertIsInstance(
            model, 
            segmentation_models_pytorch.decoders.deeplabv3.model.DeepLabV3Plus
        )

    def test_badencodername(self):
        """ Test Bad Encoder Name """
        r = RandomWord()
        word = r.word()

        self.conf.model.config.encoder_name = word

        self.assertTrue(word == self.conf.model.config.encoder_name)
        with self.assertRaises(ValueError):
            model = create_smp_model(self.conf)