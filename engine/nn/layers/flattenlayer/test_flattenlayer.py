import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/flattenlayer")
from flattenlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestFlattenLayer(unittest.TestCase):
    
    """
        Tests FlattenLayer
    """
    
    def test_constructor_full_args(self):
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
        
    def test_constructor_full_calc_args(self):
        test_name = 'flatten_1'
        test_type = FLATL_CSTYLE_TYPE
        test_mode = COLUMN_FLATTEN
        
        flatten_layer = FlattenLayer(test_name, test_type, test_mode)
        
    def test_constructor_calc_args(self):
        test_type = FLATL_CSTYLE_TYPE
        test_mode = COLUMN_FLATTEN
        
        flatten_layer = FlattenLayer(test_type, test_mode)
        
    def test_constructor_empty_args(self):
        flatten_layer = FlattenLayer()

    def test_evaluate_cstyle_layer(self):
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
        
        input = np.random.rand(5,5)
        
        result = flatten_layer.evaluate(input)
    
    def test_evaluate_nnet_layer(self):
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
        
        input = np.random.rand(5,5)
        
        result = flatten_layer.evaluate(input)
        
    def test_reach_cstyle_layer(self):
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
    
    def test_reach_nnet_layer(self):
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
        
    def test_rmi_cstyle_layer(self):
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach_multiple_inputs([test_star1, test_star1])
    
    def test_rmi_nnet_layer(self):
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_RMI_INIT][FLATL_TEST_RMI_IM_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach_multiple_inputs([test_star1, test_star1])
        
    def test_reach_cstyle_layer(self):
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_IM_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach_single_input(test_star)
    
    def test_reach_nnet_layer(self):
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
        
        test_V = np.reshape(read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_IM_LB_ID])
        test_im_ub = read_csv_data(sources[FLATL_TEST_RSI_INIT][FLATL_TEST_RSI_IM_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )

        result = flatten_layer.reach_single_input(test_star)
        
if __name__ == '__main__':
    unittest.main()