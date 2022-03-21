import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *
import numpy as np
import mat73

class TestImageStarEvaluate(unittest.TestCase):
    """
        Tests ImageStar evaluation
    """

    def test_evaluation(self):
        """
            Tests evaluation using predicate initialization
        
            eval_input : int -> number of images
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
                
        test_eval_input = self.read_csv_data(sources[EVALUATION_INIT][EVAL_INPUT_ID])
        test_eval_output = self.read_csv_data(sources[EVALUATION_INIT][EVAL_OUTPUT_ID])
                
        test_V = np.reshape(self.read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = self.read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID])
        test_d = self.read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
        test_predicate_lb = self.read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = self.read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
                
        self.assertEqual(test_star.evaluate(test_eval_input).all(), test_eval_output.all())

########################## UTILS ##########################
    def read_csv_data(self, path):        
        return np.array(list(mat73.loadmat(path).values())[0])

if __name__ == '__main__':
    unittest.main()
