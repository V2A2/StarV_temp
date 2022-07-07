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

class TestFlattenLayer_constructor(unittest.TestCase):
    
    """
        Tests FlattenLayer constructor
    """
    
    def test_full_args(self):
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
        
    def test_full_calc_args(self):
        test_name = 'flatten_1'
        test_type = FLATL_CSTYLE_TYPE
        test_mode = COLUMN_FLATTEN
        
        flatten_layer = FlattenLayer(test_name, test_type, test_mode)
        
    def test_calc_args(self):
        test_type = FLATL_CSTYLE_TYPE
        test_mode = COLUMN_FLATTEN
        
        flatten_layer = FlattenLayer(test_type, test_mode)
        
    def test_empty_args(self):
        flatten_layer = FlattenLayer()

if __name__ == '__main__':
    unittest.main()