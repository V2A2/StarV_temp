import unittest

from test_inputs.sources import *

# sys.path.insert(0, "engine/set/imagestar/")
# from imagestar import *
#
# sys.path.insert(0, "tests/test_utils/")
# from utils import *


class TestImageStarConstructor(unittest.TestCase):
    """
        Tests ImageStar constructor
    """

    def test_predicate_boundaries_init(self):
        """
            Tests the initialization with:
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
    
        completion_flag = True
    
        test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
    
        try:
            test_star = ImageStar(
                    test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
                )
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)
            
        
    def test_image_init(self):
        """
            Tests the initialization with:
            IM -> ImageStar
            LB -> Lower image
            UB -> Upper image
        """
        completion_flag = True
        
        test_IM = np.zeros((4, 4, 3))
        test_LB = np.zeros((4, 4, 3))
        test_UB = np.zeros((4, 4, 3))
    
    
        test_IM[:,:,0] = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
        test_IM[:,:,1] = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]])
        test_IM[:,:,2] = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
    
        test_LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) # attack on pixel (1,,1,) and (1,,2)
        test_LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_LB[:,:,2] = test_LB[:,:,1]
    
        test_UB[:,:,0] = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,2] = test_UB[:,:,1]
    
        try:
            test_star = ImageStar(
                    test_IM, test_LB, test_UB
                )
        except Exception as ex:
            process_exception(ex)
            completion_flag = False
            
        self.assertEqual(completion_flag, True)
        
    def test_bounds_init(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """
        completion_flag = True
        
        test_lb = read_csv_data(sources[CONSTRUCTOR_BOUNDS_INIT][TEST_LB_ID])
        test_ub = read_csv_data(sources[CONSTRUCTOR_BOUNDS_INIT][TEST_UB_ID])
        
    
        try:
            test_star = ImageStar(
                    test_lb, test_ub
                )
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)

if __name__ == '__main__':
    unittest.main()
