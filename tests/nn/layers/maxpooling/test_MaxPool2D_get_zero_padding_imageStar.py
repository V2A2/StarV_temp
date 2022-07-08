import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

os.chdir("../../../../")
print(os.getcwd())
sys.path.insert(0, "engine/nn/layers/maxpooling")
from maxpooling2dlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestAveragePooling2DLayer_get_zero_padding_imageStar(unittest.TestCase):
    """
        Tests MaxPooling2D layer's get_zero_padding_imageStar method
    """

    def test_zero_padding(self):
        
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

    def test_nonzero_padding(self):
        
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
        
        
if __name__ == '__main__':
    unittest.main()