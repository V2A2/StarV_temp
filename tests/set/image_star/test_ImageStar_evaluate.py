import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *
import numpy as np
import mat73

class TestImageStarEvaluate(unittest.TestCase):
    """
        Tests ImageStar constructor
    """

    def test_evaluation(self):
        """
            pred_val : int -> number of images
        """
                
        test_pred_val = self.read_csv_data(sources[EVALUATION_INIT][PRED_VAL_ID])
                
        test_V = self.read_csv_data(sources[CONSTRUCTOR_FIRST_INIT][V_ID])
        test_C = self.read_csv_data(sources[CONSTRUCTOR_FIRST_INIT][C_ID])
        test_d = self.read_csv_data(sources[CONSTRUCTOR_FIRST_INIT][D_ID])
        test_predicate_lb = self.read_csv_data(sources[CONSTRUCTOR_FIRST_INIT][PREDICATE_LB_ID])
        test_predicate_ub = self.read_csv_data(sources[CONSTRUCTOR_FIRST_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
                
        res = test_star.evaluate(test_pred_val)

    def test_image_init(self):
        """
            Tests the initialization with:
            IM -> ImageStar
            LB -> Lower image
            UB -> Upper image
        """
        test_IM = np.zeros((4, 4, 3))
        test_LB = np.zeros((4, 4, 3))
        test_UB = np.zeros((4, 4, 3))
        
        
        test_IM[:,:,0] = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
        test_IM[:,:,1] = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]])
        test_IM[:,:,2] = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
        
        test_LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) # attack on pixel (1,,1,) and (1,,2)
        test_LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_LB[:,:,2] = test_LB[:,:,1]
        
        test_UB[:,:,0] = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,2] = test_UB[:,:,1]
            
        test_star = ImageStar(
                test_IM, test_LB, test_UB
            )

########################## UTILS ##########################
    def read_csv_data(self, path):        
        return np.array(list(mat73.loadmat(path).values())[0])

if __name__ == '__main__':
    unittest.main()
