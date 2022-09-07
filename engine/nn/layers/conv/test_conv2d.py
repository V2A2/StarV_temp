import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/conv")
from conv2dlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestConv2DLayer(unittest.TestCase):
    """
        Tests Conv2D layer
    """

    def test_constructor_full_args(self):
        
        test_weights = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_OUTPUT_NAMES_ID])

        conv2d_layer = Conv2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)
        
    def test_constructor_calc_full_args(self):
        test_weights = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_STRIDE_ID])

        conv2d_layer = Conv2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, \
                                   test_stride, test_dilation_factor)
        
    def test_constructor_calc_args(self):
        test_weights = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_STRIDE_ID])

        conv2d_layer = Conv2DLayer(test_weights, test_bias, test_padding_size, \
                                   test_stride, test_dilation_factor)
        
    def test_constructor_full_weights_bias_args(self):
        test_weights = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))

        conv2d_layer = Conv2DLayer('transpose_convolutional_layer',\
                                   test_weights, test_bias)
        
    def test_constructor_weights_bias_args(self):
        test_weights = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID])
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))

        conv2d_layer = Conv2DLayer(test_weights, test_bias)
        
    def test_evaluate_basic(self):
        
        test_weights = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_WEIGHTS_ID]), (5,5,1,3))
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = np.array([0,0]).astype('int') #read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_OUTPUT_NAMES_ID])

        conv2d_layer = Conv2DLayer('convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)

        
        input = np.random.rand(1, 28, 28)
        
        conv2d_layer.evaluate(input)

    def test_rmi_mnist_star(self):                
        test_weights = np.reshape(read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_WEIGHTS_ID]), (5,5,1,3))
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = np.array([0,0]).astype('int') #read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_OUTPUT_NAMES_ID])

        conv2d_layer = Conv2DLayer('convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)
                    
        test_V = np.reshape(read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_D_ID])])
        test_predicate_lb = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[CONV2D_TEST_RMI_INIT][CONV2D_TEST_RMI_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        conv2d_layer.reach_multiple_inputs([test_star1, test_star2]) 
        
    def test_rsi_mnist_star(self):    
        test_weights = np.reshape(read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_WEIGHTS_ID]), (5,5,1,3))
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = np.array([0,0]).astype('int') #read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_OUTPUT_NAMES_ID])

        conv2d_layer = Conv2DLayer('convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)
                    
        test_V = np.reshape(read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[CONV2D_TEST_RSI_INIT][CONV2D_TEST_RSI_IM_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        conv2d_layer.reach_single_input(test_star)  

    def test_reach_mnist_star(self):                
        test_weights = np.reshape(read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_WEIGHTS_ID]), (5,5,1,3))
        test_bias = np.reshape(read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_BIASS_ID]), (1,1,3))
        test_filter_size = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_FILTER_SIZE_ID])
        test_padding_size = np.array([0,0]).astype('int') #read_csv_data(sources[CONV2D_TEST_CONSTRUCTOR_INIT][CONV2D_TEST_PADDING_SIZE_ID])
        test_dilation_factor = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_DILATION_FACTOR_ID])
        test_stride = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_STRIDE_ID])
        test_num_filters = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_NUM_FILTERS_ID])
        test_num_channels = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_NUM_CHANNELS_ID])
        test_num_inputs = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_OUTPUT_NAMES_ID])

        conv2d_layer = Conv2DLayer('convolutional_layer',\
                                   test_weights, test_bias, test_padding_size, test_stride, test_dilation_factor, \
                                   test_num_inputs, test_input_names, test_num_outputs, test_output_names, \
                                   test_num_filters, test_filter_size, test_num_channels)
                    
        test_V = np.reshape(read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_D_ID])])
        test_predicate_lb = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_IM_LB_ID])
        test_im_ub = read_csv_data(sources[CONV2D_TEST_REACH_INIT][CONV2D_TEST_REACH_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        conv2d_layer.reach([test_star1, test_star2])   
        
if __name__ == '__main__':
    unittest.main()