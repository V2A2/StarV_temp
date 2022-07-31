import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

os.chdir("../../../../")
print(os.getcwd())
sys.path.insert(0, "engine/nn/layers/signlayer")
from signlayer import *

sys.path.insert(0, "engine/set/star")
from star import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/zono")
from zono import *
    
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestSignLayer_reach(unittest.TestCase):
    
    """
        Tests reachability analysis method for multiple inputs
    """
    
    def test_star_exact(self):
        test_name = 'sign_1'
        test_mode = 'polar_zero_to_pos_one'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        sign_layer = SignLayer(test_name, test_mode, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
    
        test_V = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_V_ID])
        test_C = np.reshape(read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_D_ID])])
        test_predicate_lb = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_PREDICATE_UB_ID])
        
        test_zono_V = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_ZONO_V_ID])
        test_zono_c = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_ZONO_C_ID])
    
        test_star1 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star1.Z = Zono(test_zono_c, test_zono_V)
        
        test_star2 = Star(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        test_star2.Z = Zono(test_zono_c, test_zono_V)
    
        result = sign_layer.reach([test_star1,test_star2], 'approx-star')
    
    # def test_star_approx(self):
    #     test_name = 'sign_1'
    #     test_mode = 'polar_zero_to_pos_one'
    #     test_num_inputs = 1
    #     test_input_names = ['in']
    #     test_num_outputs = 1
    #     test_output_names = ['out']
    #
    #     sign_layer = SignLayer(test_name, test_mode, test_num_inputs,\
    #                               test_input_names, test_num_outputs,\
    #                               test_output_names)
    #
    #     test_V = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_V_ID])
    #     test_C = np.reshape(read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_D_ID])])
    #     test_predicate_lb = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_STAR_PREDICATE_UB_ID])
    #
    #     test_star = Star(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    #
    #     result = sign_layer.reach_star_single_input(test_star, 'approx-star')
    #
    # def test_imagestar(self):
    #     test_name = 'relu_1'
    #     test_num_inputs = 1
    #     test_input_names = ['in']
    #     test_num_outputs = 1
    #     test_output_names = ['out']
    #
    #     relu_layer = ReLULayer(test_name, test_num_inputs,\
    #                               test_input_names, test_num_outputs,\
    #                               test_output_names)
    #
    #     test_V = np.reshape(read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_C_ID]), (1, 784))
    #     test_d = np.array([read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_D_ID])])
    #     test_predicate_lb = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_PREDICATE_UB_ID])
    #     test_im_lb = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_IM_LB_ID])
    #     test_im_ub = read_csv_data(sources[SIGNL_TEST_REACH_INIT][SIGNL_TEST_REACH_IMGSTAR_IM_UB_ID])
    #
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
    #         )
    #
    #     result = relu_layer.reach_star_single_input(test_star, 'approx-star')

if __name__ == '__main__':
    unittest.main()