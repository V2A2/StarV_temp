import unittest

from test_inputs.sources import *

class TestImageStarGetLocalMaxIndex(unittest.TestCase):
    """
        Tests the 'get_localMax_index' method
        
        TODO: CONFIRM get_local_points and retest
    """

    def test_get_local_max_index_empty_candidates(self):
        """
            Tests the ImageStar's method that calculates the local maximum point of the local image.
            Candidates set will be empty 
    
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
    
        test_result = test_star.get_localMax_index(test_local_index_input[0:2] - 1, test_local_index_input[2:4], test_local_index_input[4] - 1)
    
    
        self.assertEqual(test_result, test_local_index_output)

    def test_get_local_max_index_candidates(self):
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
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_CANDIDATES_ID]), (24, 24, 3, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_CANDIDATES_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_CANDIDATES_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_CANDIDATES_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_CANDIDATES_ID])
        test_im_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_LB_CANDIDATES_ID])
        test_im_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_UB_CANDIDATES_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
               
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_CANDIDATES_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_CANDIDATES_ID])).tolist()
                        
        test_result = test_star.get_localMax_index(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4])
                                
        self.assertEqual(test_result, test_local_index_output)


if __name__ == '__main__':
    unittest.main()
