import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarIsP1LargerThanP2(unittest.TestCase):
    """
        Tests is_p1_larger_than_p2() function. This function checks if an index of a point in Star is larger than an index of other point.
    """
    
    def test_is_p1_larger_than_p2(self):
        """
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                True -> if there exists the case that p1 >= p2
                False  -> if there is no case that p1 >= p2
        """
        
        lb = np.array([-3, 4, -1])
        ub = np.array([3, 6, 3])
    
        S = Star(lb, ub)
        print('Is p1 larger than p2? ', S.is_p1_larger_than_p2(0, 1))
        print('Is p2 larger than p1? ', S.is_p1_larger_than_p2(1, 0))
        print('Is p2 larger than p3? ', S.is_p1_larger_than_p2(1, 2))  
        print('Is p3 larger than p2? ', S.is_p1_larger_than_p2(2, 1))  
   
if __name__ == '__main__':
    unittest.main()