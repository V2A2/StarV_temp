import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/sigmoidlayer")
from sigmoidlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestSigmoidLayer(unittest.TestCase):
    
    """
        Tests SigmoidLayer
    """
    
    def test_constructor_full_args(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
        
    def test_constructor_name_args(self):
        test_name = 'sigmoid_1'
        
        sigmoid_layer = SigmoidLayer(test_name)
        
    def test_constructor_empty_args(self):
        sigmoid_layer = SigmoidLayer()
        
    def test_evaluate_cstyle_layer(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
        
        input = np.random.rand(3,3) - 0.5

        result = sigmoid_layer.evaluate(input)

    def test_reach_imagestar(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
    
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
    
        test_V = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_LB_ID])
        test_im_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_UB_ID])
    
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        result = sigmoid_layer.reach([test_star1, test_star2], 'approx-star')
        
    def test_rmi_imagestar(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
    
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
    
        test_V = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_LB_ID])
        test_im_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_UB_ID])
    
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        result = sigmoid_layer.reach_star_multiple_inputs([test_star1, test_star2], 'approx-star')

    def test_rsi_imagestar(self):
        test_name = 'sigmoid_1'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
    
        sigmoid_layer = SigmoidLayer(test_name, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
    
        test_V = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_LB_ID])
        test_im_ub = read_csv_data(sources[SIGMOIDL_TEST_RSI_INIT][SIGMOIDL_TEST_RSI_IMGSTAR_IM_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
    
        result = sigmoid_layer.reach_star_single_input(test_star, 'approx-star')

if __name__ == '__main__':
    unittest.main()