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

class TestFlattenLayer_reach(unittest.TestCase):
    
    """
        Tests reachability analysis method for multiple inputs
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach([test_star1, test_star1])
    
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_REACH_INIT][FLATL_TEST_REACH_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach([test_star1, test_star1])

if __name__ == '__main__':
    unittest.main()