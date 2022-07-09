import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "engine/nn/layers/maxpooling")
from maxpooling2dlayer import *

sys.path.insert(0, "tests/nn/layers/maxpooling")
from test_inputs.sources import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestAveragePooling2DLayer_get_size_max_map(unittest.TestCase):
    """
        Tests MaxPooling2D layer's get_size_maxMap method
    """

    def test_basic(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT][MAXP2D_TEST_GET_SIZE_MAX_MAP_V_ID]), (24,24,3,785))
                
        maxpool2d_layer.get_size_maxMap(test_V[:,:,0,0], maxpool2d_layer.get_zero_padding_input(test_V[:,:,0,0]))
        
        
if __name__ == '__main__':
    unittest.main()