import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "engine/nn/layers/maxpooling")
from maxpooling2dlayer import *

from test_inputs.sources import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestMaxPool2DLayer(unittest.TestCase):
    """
        Tests MaxPooling2D layer
    """

    def test_constructor_full_args(self):
        
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

    def test_constructor_full_calc_args(self):
    
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
    
        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size)
    
    def test_constructor_calc_args(self):
    
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_CONSTRUCTOR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
    
        maxpool2d_layer = MaxPooling2DLayer(test_pool_size, test_stride, test_padding_size)

    def test_evaluate_basic(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_EVALUATE_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        input = np.random.rand(1, 28, 28)
        
        maxpool2d_layer.evaluate(input)
        
    def test_get_size_maxMap_basic(self):
        
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

    def test_get_start_points_basic(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_START_POINTS_INIT][MAXP2D_TEST_GET_START_POINTS_V_ID]), (24,24,3,785))
                
        maxpool2d_layer.get_start_points(test_V[:,:,0,0])
        
    def test_get_zero_padding_imageStar_zero_padding(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        maxpool2d_layer.get_zero_padding_imageStar(test_star)

    def test_get_zero_padding_imageStar_nonzero_padding(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2)) + 1
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT][MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        maxpool2d_layer.get_zero_padding_imageStar(test_star)
        
    def test_get_zero_padding_input_basic(self):
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)

        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT][MAXP2D_TEST_GET_ZERO_PADDING_INPUT_V_ID]), (24,24,3,785))
                
        maxpool2d_layer.get_zero_padding_input(test_V[:,:,0,0])
        
    def test_reach_star_approx_multiple_inputs_basic(self):
                
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_RMI_APPROX_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_RMI_APPROX_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_RMI_APPROX_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_RMI_APPROX_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_RMI_APPROX_INIT][MAXP2D_TEST_RMI_APPROX_PREDICATE_UB_ID])
                
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
                
        maxpool2d_layer.reach_star_approx_multiple_inputs([test_star1, test_star2])

    def test_reach_approx_single_input_basic(self):
        
        #raise NotImplementedError
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_RSI_APPROX_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_RSI_APPROX_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_RSI_APPROX_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_RSI_APPROX_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_RSI_APPROX_INIT][MAXP2D_TEST_RSI_APPROX_PREDICATE_UB_ID])
                
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
                
        maxpool2d_layer.reach_approx_single_input(test_star)
        
    def test_reach_exact_multiple_inputs_basic(self):
        
        raise NotImplementedError #TODO: change to halting to prevent 'OutOfMemory exception'
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_RMI_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_RMI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_RMI_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_RMI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_RMI_INIT][MAXP2D_TEST_RMI_PREDICATE_UB_ID])
                
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
                
        maxpool2d_layer.reach_exact_multiple_inputs([test_star1, test_star2])

    def test_reach_exact_multiple_inputs_basic(self):
        
        #raise NotImplementedError
        
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_RSI_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_RSI_INIT][MAXP2D_TEST_RSI_PREDICATE_UB_ID])
                
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
                
        maxpool2d_layer.reach_exact_single_input(test_star)

    def test_reach_zono_multiple_inputs_basic(self):
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RMI_ZONO_INIT][MAXP2D_TEST_RMI_ZONO_V_ID]), (24,24,3,785))
        
        test_zono1 = ImageZono(test_V)
        test_zono2 = ImageZono(test_V)

        
        maxpool2d_layer.reach_zono_multiple_inputs([test_zono1, test_zono2])

    def test_reach_zono_single_input_basic(self):
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_RSI_ZONO_INIT][MAXP2D_TEST_RSI_ZONO_V_ID]), (24,24,3,785))
        
        test_zono = ImageZono(test_V)
                
        maxpool2d_layer.reach_zono_single_input(test_zono)
        
    def test_star_exact(self):  
        raise NotImplementedError #TODO: change to halting to prevent 'OutOfMemory exception'
    
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])
    
        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
    
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_PREDICATE_UB_ID])
    
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
    
        maxpool2d_layer.reach([test_star1, test_star2], 'exact-star')

    def test_star_approx(self):  
              
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_V_ID]), (24,24,3,785))
        test_C = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_D_ID])])
        test_predicate_lb = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_PREDICATE_UB_ID])
                
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )   
                
        maxpool2d_layer.reach([test_star1, test_star2], 'approx-star', [])

    def test_zono_approx(self):  
              
        test_pool_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_POOL_SIZE_ID]), (1,2))
        test_padding_size = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_PADDING_SIZE_ID]), (1, 2))
        test_stride = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_STRIDE_ID]), (1, 2))
        test_num_inputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_INPUTS_ID])
        test_input_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_INPUT_NAMES_ID])
        test_num_outputs = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_NUM_OUTPUTS_ID])
        test_output_names = read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_OUTPUT_NAMES_ID])

        maxpool2d_layer = MaxPooling2DLayer('max_pooling_2d_layer',\
                                   test_pool_size, test_stride, test_padding_size, test_num_inputs, \
                                   test_input_names, test_num_outputs, test_output_names)
        
        test_V = np.reshape(read_csv_data(sources[MAXP2D_TEST_REACH_INIT][MAXP2D_TEST_REACH_V_ID]), (24,24,3,785))

        test_zono1 = ImageZono(test_V)
        test_zono2 = ImageZono(test_V)
                
        maxpool2d_layer.reach([test_zono1, test_zono2], 'approx-zono', [])

        
if __name__ == '__main__':
    unittest.main()