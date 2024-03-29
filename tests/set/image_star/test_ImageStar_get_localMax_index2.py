import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarGetLocalMaxIndex2(unittest.TestCase):
    """
        Tests the 'get_localMax_index2' method used for overapproximating max pooling
        
        TODO: CONFIRM get_local_points and retest
    """ 
    
    def test_get_local_max_index2_candidates(self):
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
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
               
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][INPUT_CANDIDATES_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][OUTPUT_CANDIDATES_ID])  - 1).tolist()
                                
        self.assertEqual(test_star.get_localMax_index2(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4]), test_local_index_output)


if __name__ == '__main__':
    unittest.main()
