import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarIsP1LargerP2(unittest.TestCase):
    """
        Tests the 'is_p1_larger_p2' method
    """

    def test_is_p1_larger_p2(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> valid input
            local_index -> valid output bounds
        """
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_points_input = np.array([int(item) for item in read_csv_data(sources[IS_P1_LARGER_P2_INIT][INPUT_ID])])
        test_points_output = (read_csv_data(sources[IS_P1_LARGER_P2_INIT][OUTPUT_ID])  - 1).tolist()
    

        self.assertEqual(test_star.is_p1_larger_p2(test_points_input[0:3], test_points_input[3:6]), test_points_output)

if __name__ == '__main__':
    unittest.main()
