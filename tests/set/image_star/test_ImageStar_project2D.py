import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarProject2D(unittest.TestCase):
    """
        Tests the 'project2D' method
    """

    def test_contains_true(self):
        """
            Tests the ImageStar's projection on the given plain formulated by two points
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_point1 -> the first point
            test_point2 -> the second point
        """
        
        test_V = np.reshape(read_csv_data(sources[PROJECT2D_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[PROJECT2D_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[PROJECT2D_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_point1 = np.array([5,5,1])#read_csv_data(sources[PROJECT2D_INIT][POINT1_ID])
        test_point2 = np.array([4,7,1])#read_csv_data(sources[PROJECT2D_INIT][POINT2_ID])
        
        self.assertEqual(test_star.project2D(test_point1, test_point2), True)

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
