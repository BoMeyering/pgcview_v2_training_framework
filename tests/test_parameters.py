"""
tests/test_parameters.py
Tests the parameters.py functions and classes
BoMeyering 2025
"""

import unittest
from src.parameters import OptimConfig, EMA, apply_ema

class TestOptimConfig(unittest.TestCase):
    """ Unit Test the OptimConfig class """
    def setUp(self):
        return super().setUp()
    
    def test_init(self):
        pass

    def test_add_weight_decay(self):
        pass

    def test_iter_trainable_params(self):
        pass

    def test_get_params(self):
        pass

    def test_get_optimizer(self):
        pass

    def test_get_scheduler(self):
        pass

    def test_process(self):
        pass

class TestEMA(unittest.TestCase):
    """ Unit Test the EMA class """
    def setUp(self):
        return super().setUp()
    
    def test_init(self):
        pass

    def test_update_params(self):
        pass

    def test_assign_params(self):
        pass

    def test_update_and_assign_params(self):
        pass

    def test_restore_params(self):
        pass

    def test_apply_ema_context_manager(self):
        pass