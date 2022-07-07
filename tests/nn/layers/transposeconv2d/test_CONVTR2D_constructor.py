import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/transposeconv2d")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/transposeconv")
from transposeconv2dlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestConvTr2DLayer_constructor(unittest.TestCase):
    """
        Tests transposed Conv2D layer constructor
    """

    def test_full_args(self):
        
        test_weights = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_OUTPUT_NAMES_ID])

        trconv2d_layer = ConvTranspose2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)
        
    def test_calc_full_args(self):
        test_weights = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_STRIDE_ID])
    
        trconv2d_layer = ConvTranspose2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, \
                                   test_stride, test_dilation_factor)
    
    def test_calc_args(self):
        test_weights = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_STRIDE_ID])
    
        trconv2d_layer = ConvTranspose2DLayer(test_weights, test_bias, test_padding_size, \
                                   test_stride, test_dilation_factor)
    
    def test_full_weights_bias_args(self):
        test_weights = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_BIASS_ID]), (1,1,3))
    
        trconv2d_layer = ConvTranspose2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias)
    
    def test_weights_bias_args(self):
        test_weights = read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONVTR2D_TEST_CONSTRUCTOR_INIT][CONVTR2D_TEST_BIASS_ID]), (1,1,3))
    
        trconv2d_layer = ConvTranspose2DLayer(test_weights, test_bias)

        
if __name__ == '__main__':
    unittest.main()