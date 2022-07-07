import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/flatten")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/flattenlayer")
from flattenlayer import *

def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestFlattenLayer_evaluate(unittest.TestCase):
    
    """
        Tests FlattenLayer evaluate method
    """
    
    def test_cstyle_layer(self):
        test_name = 'flatten_1'
        test_type = FLATL_CSTYLE_TYPE
        test_mode = COLUMN_FLATTEN
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        flatten_layer = FlattenLayer(test_name, test_mode,\
                                     test_type, test_num_inputs,\
                                     test_input_names, test_num_outputs,\
                                     test_output_names)
        
        input = np.random.rand(5,5)
        
        result = flatten_layer.evaluate(input)
    
    def test_nnet_layer(self):
        test_name = 'flatten_1'
        test_type = FLATL_NNET_TYPE
        test_mode = COLUMN_FLATTEN
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        flatten_layer = FlattenLayer(test_name, test_mode,\
                                     test_type, test_num_inputs,\
                                     test_input_names, test_num_outputs,\
                                     test_output_names)
        
        input = np.random.rand(5,5)
        
        result = flatten_layer.evaluate(input)
        
if __name__ == '__main__':
    unittest.main()