import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar/")

from imagestar import *

class TestImageStarEstimateRange(unittest.TestCase):
    """
        Tests the 'estimate_range' method
    """

    def test_estimate_range(self):
        """
            Tests the ImageStar's range estimation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_input -> valid input range
            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[ESTIMATE_RANGE_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[ESTIMATE_RANGE_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[ESTIMATE_RANGE_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[ESTIMATE_RANGE_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[ESTIMATE_RANGE_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        range_input = np.array(read_csv_data(sources[ESTIMATE_RANGE_INIT][INPUT_ID]))
        range_output = np.array([read_csv_data(sources[ESTIMATE_RANGE_INIT][ESTIMATE_RANGE_OUTPUT_ID])])
        
        self.assertEqual(test_star.estimate_range(range_input[VERT_ID], range_input[HORIZ_ID], range_input[CHANNEL_ID]).all(), range_output.all())

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
