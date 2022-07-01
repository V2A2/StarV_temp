import unittest
import numpy as np
import sys, os

sys.path.insert(0, "tests/nn/layers/fullyconnected")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/fullyconnected")
from fullyconnectedlayer import *

class TestFCLayer_constructor(unittest.TestCase):
    """
        Tests Fullyconnected layer constructor
    """

    def test_full_args(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
    def test_calc_args(self):
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(W, b)
    
    def test_empty_args(self):
        fc_layer = FullyConnectedLayer()
    
    def test_invalid_len_args(self):
        q = 0
        # TODO: catch the exception
        fc_layer = FullyConnectedLayer(q)
        
        
        
if __name__ == '__main__':
    unittest.main()