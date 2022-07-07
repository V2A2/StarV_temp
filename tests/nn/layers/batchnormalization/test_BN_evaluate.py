import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/batchnormalizationlayer")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/batchnormalization")
from batchnormalizationlayer import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestBNLayer_evalate(unittest.TestCase):
    """
        Tests Batchnormalization layer's evaluate method
    """

    def test_basic(self):
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
        
        
if __name__ == '__main__':
    unittest.main()