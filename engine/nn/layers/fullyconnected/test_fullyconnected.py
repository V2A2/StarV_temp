import unittest
import numpy as np
import sys, os

from test_inputs.sources import *

sys.path.insert(0, "engine/nn/layers/fullyconnected")
from fullyconnectedlayer import *

class TestFCLayer(unittest.TestCase):
    """
        Tests Fullyconnected layer constructor
    """

    def test_constructor_full_args(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
    def test_constructor_calc_args(self):
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(W, b)
    
    def test_constructor_empty_args(self):
        fc_layer = FullyConnectedLayer()
    
    def test_constructor_invalid_len_args(self):
        q = 0
        # TODO: catch the exception
        fc_layer = FullyConnectedLayer(q)
        
    def test_reach_basic(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        input = np.random.rand(4, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        fc_layer.evaluate(input)
    
    def test_reach_invalid_dims(self):
        name = 'FC_test1'
        W = np.random.rand(4, 4)
        b = np.random.rand(4, 1)
        
        input = np.random.rand(3, 1)
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        #TODO: catch the exception here
        fc_layer.evaluate(input)
        
    def test_reach_mnist_star(self):                
        test_V = np.reshape(read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][V_ID]), (28, 28, 1,785))
        test_C = np.reshape(read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][PREDICATE_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        name = 'FC_test1'
        W = read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][FC_REACH_MULTIPLE_INPUTS_WEIGHTS_ID])
        b = read_csv_data(sources[FC_REACH_MULTIPLE_INPUTS_INIT][FC_REACH_MULTIPLE_INPUTS_BIAS_ID])
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        fc_layer.reach_multiple_inputs([test_star1, test_star2])

    def test_reach_mnist_imagestar(self):     
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

    def test_reach_mnist_star(self):                
        test_V = np.reshape(read_csv_data(sources[FC_REACH_INIT][V_ID]), (28, 28, 1,785))
        test_C = np.reshape(read_csv_data(sources[FC_REACH_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[FC_REACH_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[FC_REACH_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[FC_REACH_INIT][PREDICATE_UB_ID])
        
        test_star1 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_star2 = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        name = 'FC_test1'
        W = read_csv_data(sources[FC_REACH_INIT][FC_REACH_WEIGHTS_ID])
        b = read_csv_data(sources[FC_REACH_INIT][FC_REACH_BIAS_ID])
        
        fc_layer = FullyConnectedLayer(name, W, b)
        
        fc_layer.reach([test_star1, test_star2])

        
if __name__ == '__main__':
    unittest.main()