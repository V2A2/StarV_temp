import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/batchnormalization")
from batchnormalizationlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestBNLayer(unittest.TestCase):
    """
        Tests Batchnormalization layer
    """

    def test_constructor_valid_args(self):
        
        iput_dict = {
            "Name" : 'BN_test1',
            'NumChannels' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_CHANNELS_ID]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)
    

    def test_constructor_invalid_args(self):
        
        iput_dict = {
            "qwerty" : 'BN_test1',
            'NumChannels' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_CHANNELS_ID]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_CONSTRUCTOR_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)            
        
    def test_evaluate_basic(self):
        iput_dict = {
            "Name" : 'BN_test1',
            'NumChannels' : int(np.array([read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_NUM_CHANNELS_ID])])[0]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_EVALUATE_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)
        
        input = np.random.rand(1, 784)
        
        bn_layer.evaluate(input)

    def test_rmi_mnist_star(self):                
        iput_dict = {
            "Name" : 'BN_test1',
            'NumChannels' : int(np.array([read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_NUM_CHANNELS_ID])])[0]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)
                    
        test_V = np.reshape(read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_RSI_V_ID]), (784,785))
        test_C = np.reshape(read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[BN_TEST_REACH_MULTIPLE_INPUTS_INIT][BN_TEST_RSI_PREDICATE_UB_ID])
        
        test_star1 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_star2 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        bn_layer.reach_multiple_inputs([test_star1, test_star2])
        
    def test_rsi_mnist_star(self):    
        iput_dict = {
            "Name" : 'BN_test1',
            'NumChannels' : int(np.array([read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_NUM_CHANNELS_ID])])[0]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)
                    
        test_V = np.reshape(read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_RSI_V_ID]), (784,785))
        test_C = np.reshape(read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[BN_TEST_REACH_SINGLE_INPUT_INIT][BN_TEST_RSI_PREDICATE_UB_ID])
        
        test_star = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        bn_layer.reach_single_input(test_star)
        
    def test_reach_mnist_star(self):                
        iput_dict = {
            "Name" : 'BN_test1',
            'NumChannels' : int(np.array([read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_NUM_CHANNELS_ID])])[0]),
            'TrainedMean' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_TRAINED_MEAN_ID]),
            'TrainedVariance' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_TRAINED_VAR_ID]),
            'Epsilon' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_EPSILON_ID]),
            'Offset' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_OFFSET_ID]),
            'Scale' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_SCALE_ID]),
            'NumInputs' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_NUM_INPUTS_ID]),
            'InputNames' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_INPUT_NAMES_ID]),
            'NumOutputs' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_NUM_OUTPUTS_ID]),
            'OutputNames' : read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_OUTPUT_NAMES_ID])
        }
        
        bn_layer = BatchNormalizationLayer(iput_dict)
                    
        test_V = np.reshape(read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_RSI_V_ID]), (784,785))
        test_C = np.reshape(read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_RSI_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_RSI_D_ID])])
        test_predicate_lb = read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_RSI_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[BN_TEST_REACH_INIT][BN_TEST_RSI_PREDICATE_UB_ID])
        
        test_star1 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_star2 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        bn_layer.reach_multiple_inputs([test_star1, test_star2])
        
if __name__ == '__main__':
    unittest.main()