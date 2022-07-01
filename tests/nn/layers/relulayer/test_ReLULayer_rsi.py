import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/relulayer")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/relulayer")
from relulayer import *

sys.path.insert(0, "engine/set/star")
from star import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestReLULayer_rsi(unittest.TestCase):
    
    """
        Tests reachability analysis method for a single input
    """
    
    # def test_star(self):
    #     test_name = 'relu_1'
    #     test_num_inputs = 1
    #     test_input_names = ['in']
    #     test_num_outputs = 1
    #     test_output_names = ['out']
    #
    #     relu_layer = ReLULayer(test_name, test_num_inputs,\
    #                               test_input_names, test_num_outputs,\
    #                               test_output_names)
    #
    #     test_V = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_STAR_V_ID])
    #     test_C = np.reshape(read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_STAR_C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_STAR_D_ID])])
    #     test_predicate_lb = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_STAR_PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_STAR_PREDICATE_UB_ID])
    #
    #     test_star = Star(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    #
    #     result = relu_layer.reach_star_single_input(test_star, 'approx-star')
    
    def test_imagestar(self):
        test_name = 'relu_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
    
        relu_layer = ReLULayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
    
        test_V = np.reshape(read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_IM_LB_ID])
        test_im_ub = read_csv_data(sources[RELUL_TEST_RSI_INIT][RELUL_TEST_RSI_IMGSTAR_IM_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        result = relu_layer.reach_star_single_input(test_star, 'approx-star')

if __name__ == '__main__':
    unittest.main()