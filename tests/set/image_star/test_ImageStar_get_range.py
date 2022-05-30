import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar/")

from imagestar import *

class TestImageStarGetRange(unittest.TestCase):
    """
        Tests the 'get_range' method
    """

    def test_contains_true(self):
        """
            Tests the ImageStar's range method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            range_input -> input
            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[GETRANGE_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[GETRANGE_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GETRANGE_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GETRANGE_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GETRANGE_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        range_input = np.array([int(item) for item in read_csv_data(sources[GETRANGE_INIT][INPUT_ID])])
        range_output = np.array([read_csv_data(sources[GETRANGE_INIT][OUTPUT_ID])])
        
        test_result = test_star.get_range(*range_input)
        
        self.assertEqual(test_result.all(), range_output.all())

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
