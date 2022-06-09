import unittest

from test_inputs.sources import *

class TestImageStarAddConstraints(unittest.TestCase):
    """
        Tests the 'add_constraints' method
    """

    def test_add_constraints(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> new constraints
        """
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_C, test_d = ImageStar.add_constraints(test_star, [4, 4, 0], [1,1,0])
        
        

if __name__ == '__main__':
    unittest.main()
