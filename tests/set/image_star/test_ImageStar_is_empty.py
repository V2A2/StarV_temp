import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar/")
from imagestar import *

sys.path.insert(0, "../../../tests/test_utils/")
from utils import *

class TestImageStarIsEmpty(unittest.TestCase):
    """
        Tests the 'is_empty' method
    """

    def test_is_empty_false(self):
        """
            Checks if the initialized ImageStar is empty
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
        
        test_V = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_EMPTY_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        completion_flag = True
        
        try:
            test_result = test_star.is_empty_set()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(completion_flag, True)

    def test_is_empty_true(self):
        """
            Checks if the empty ImageStar is empty
        """
        
        test_star = ImageStar()
        
        completion_flag = True
        
        try:
            test_result = test_star.is_empty_set()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(completion_flag, True)

if __name__ == '__main__':
    unittest.main()
