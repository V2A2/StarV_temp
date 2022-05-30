import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar/")

from imagestar import *

class TestImageStarUpdateRanges(unittest.TestCase):
    """
        Tests the 'update_ranges' method
    """

    def test_update_ranges(self):
        """
            Tests the ImageStar's ranges calculation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[UPDATE_RANGES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        ranges_input = [np.array([0, 0, 0])]
        
        ranges_output = np.array([read_csv_data(sources[UPDATE_RANGES_INIT][UPDATERANGES_OUTPUT_ID])])
                
        ranges = test_star.update_ranges(ranges_input)
                
        res_flag = True
        
        for i in range(len(ranges)):
            if ranges[i].all() != ranges_output[i].all():
                res_flag = False
                break
                
        self.assertEqual(res_flag, True)

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
