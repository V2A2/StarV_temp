import unittest
import numpy as np
import sys, os

sys.path.insert(0, "tests/nn/layers/fullyconnected")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/fullyconnected")
from fullyconnectedlayer import *

class TestFCLayer_evalate(unittest.TestCase):
    """
        Tests Fullyconnected layer's evaluate method
    """

    def test_basic(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        input = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        fc_layer.evaluate(input)
    
    def test_invalid_dims(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        input = np.random.rand(3, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        #TODO: catch the exception here
        fc_layer.evaluate(input)
        
    # TODO: add a real-world test (similar to ImageStar)
        
if __name__ == '__main__':
    unittest.main()