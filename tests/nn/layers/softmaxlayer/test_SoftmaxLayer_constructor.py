import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/sigmoidlayer")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/sigmoidlayer")
from sigmoidlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestSigmoidLayer_constructor(unittest.TestCase):
    
    """
        Tests SigmoidLayer constructor
    """
    
    def test_full_args(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
        
    def test_name_args(self):
        test_name = 'sigmoid_1'
        
        sigmoid_layer = SigmoidLayer(test_name)
        
    def test_empty_args(self):
        sigmoid_layer = SigmoidLayer()

if __name__ == '__main__':
    unittest.main()