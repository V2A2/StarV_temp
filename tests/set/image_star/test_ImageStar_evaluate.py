import unittest

from test_inputs.sources import *

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
                
        test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
                
        try:
            test_result = test_star.evaluate(test_eval_input)
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
                
        self.assertEqual(test_result.all(), test_eval_output.all())

########################## UTILS ##########################
    def read_csv_data(self, path):        
        return np.array(list(mat73.loadmat(path).values())[0])

if __name__ == '__main__':
    unittest.main()
