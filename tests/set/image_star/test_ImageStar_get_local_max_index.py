import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarGetLocalMaxIndex(unittest.TestCase):
    """
        Tests the 'get_localMax_index' method
    """

    def test_get_local_max_index(self):
        """
            Tests the ImageStar's method that calculates the local maximum point of the local image. 
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            local_index -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_ID])  - 1).tolist()
                                
        self.assertEqual(test_star.get_localMax_index(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4]), test_local_index_output)

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
