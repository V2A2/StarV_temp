import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar")

from imagestar import *

class TestImageStarGetLocalBound(unittest.TestCase):
    """
        Tests the 'get_local_bound' method
    """

    def test_get_local_bound(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4]), test_bounds_output)

    # def test_contains_fase(self):
    #     """
    #         Checks if the initialized ImageStar is empty
    #
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    #     """
    #
    #     test_V = np.reshape(read_csv_data(sources[CONTAINS_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[CONTAINS_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[CONTAINS_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[CONTAINS_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[CONTAINS_INIT][PREDICATE_UB_ID])
    #
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    #
    #     test_input = read_csv_data(sources[CONTAINS_INIT][FALSE_INPUT])
    #
    #     self.assertEqual(test_star.contains(test_input), False)

if __name__ == '__main__':
    unittest.main()
