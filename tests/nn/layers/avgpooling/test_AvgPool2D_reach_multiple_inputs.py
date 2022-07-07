import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/avgpooling")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/avgpooling")
from averagepooling2dlayer import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *
    
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestConv2DLayer_reach_multiple_inputs(unittest.TestCase):
    """
        Tests Conv2D layer's reachability method for multiple inputs
    """

    def test_fmnist_imagestar(self):    
        test_pool_size = np.reshape(read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_OUTPUT_NAMES_ID])

        avgpool2d_layer = AveragePooling2DLayer('average_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
                    
        test_V = np.reshape(read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_RMI_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_RMI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_RMI_D_ID])])
        test_predicate_lb = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_RMI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[AVGP2D_TEST_RMI_INIT][AVGP2D_TEST_RMI_PREDICATE_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        avgpool2d_layer.reach_multiple_inputs([test_star1, test_star2])
    # TODO: add ImageStar, Zono, ImageZono tests
        
if __name__ == '__main__':
    unittest.main()
