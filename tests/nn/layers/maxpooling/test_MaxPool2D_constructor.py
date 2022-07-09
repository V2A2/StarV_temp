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

class TestMaxPool2DLayer_constructor(unittest.TestCase):
    """
        Tests MaxPooling2D layer constructor
    """

    def test_full_args(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

    def test_full_calc_args(self):
    
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
    
        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size)
    
    def test_calc_args(self):
    
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
    
        maxpool2d_layer = MaxPooling2DLayer(test_pool_size, test_stride, test_padding_size)
        

        
if __name__ == '__main__':
    unittest.main()