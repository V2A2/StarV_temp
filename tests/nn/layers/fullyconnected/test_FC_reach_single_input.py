import unittest
import numpy as np
import sys, os
import mat73

sys.path.insert(0, "tests/nn/layers/fullyconnected")
from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/fullyconnected")
from fullyconnectedlayer import *

sys.path.insert(0, "engine/set/star")
from star import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestFCLayer_reach_single_input(unittest.TestCase):
    """
        Tests Fullyconnected layer's reachability method for a single input
    """

    def test_mnist_imagestar(self):     
        # TODO: check this one with the imagestar      
        test_V = np.reshape(read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        name = 'FC_test1'
        W = read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][FC_REACH_SINGLE_INPUT_WEIGHTS_ID])
        b = read_csv_data(sources[FC_REACH_SINGLE_INPUT_INIT][FC_REACH_SINGLE_INPUT_BIAS_ID])
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        fc_layer.reach_single_input(test_star)
        
if __name__ == '__main__':
    unittest.main()
