import unittest

from test_inputs.sources import *

class TestImageStarAttackedPixelsNum(unittest.TestCase):
    """
        Tests the 'get_num_attacked_pixels' method
    """

    def test_num_attacked_pixels(self):
        """
            Tests the ImageStar's method that calculates the number of attacked pixels
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        attacked_pixels_num_output = np.array([read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][ATTACKPIXNUM_OUTPUT_ID])])
                                
        test_result = test_star.get_num_attacked_pixels()
                                
        self.assertEqual(test_result, attacked_pixels_num_output)

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
