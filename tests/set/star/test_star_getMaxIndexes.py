import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarGetMaxIndexes(unittest.TestCase):
    """
        Tests getMaxIndexes() function that returns possible max indexes
    """

    def test_getMaxIndexes(self):
        """
            Randomely generate star set by providing 
            dim : dimension of a star to be generated
            N : maximum number of constraints to be generated on a star
                
            Output -> index of the state that can be a max point          
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        # S = Star.rand(dim, N)
        # print('\nPrint generated star in detail: \n')
        # print(S.__repr__())
        
        V1 = np.array([[0, 1, 0], [0, 0, 1]])
        C1 = np.array([[0.99985, 0.01706], [-0.40967, 0.91224], [-0.57369, -0.81907]])
        d1 = np.array([0.95441, 0.46047, -0.82643])
        predicate_lb_1 = np.array([0.43863, 0.34452])
        predicate_ub_1 = np.array([0.94867, 0.92634])
        S1 = Star(V1, C1, d1, predicate_lb_1, predicate_ub_1)
        print('\nPrint S1 in detail: \n')
        print(S1.__repr__())
        S1.plot()
        
   
        max_id = S1.getMaxIndexes()
        print('\n Possible max indexes: ', max_id)
    
if __name__ == '__main__':
    unittest.main()