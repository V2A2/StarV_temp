import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

#------------To Do: Need to fix this function ----------#
class TestZonoP1LargerThanP2(unittest.TestCase):
    """
        Tests if an index of a point in Zono is larger than an index of other point 
    """
    
    def test_is_p1_larger_than_p2(self):
        """
            Tests with initializing Box with lower bound and upper bound vectors and
            coverting it to zonotope. 
                
            return:
                True -> if there exists the case that p1 >= p2
                False -> if there is no case that p1 >= p2
        """
        lb = np.array([-3, 4, -1])
        ub = np.array([3, 6, 3])
        B = Box(lb, ub)
        Z = B.toZono()
        
        print('Is p1 larger than p2? ', Z.is_p1_larger_than_p2(0, 1))
        print('Is p2 larger than p1? ', Z.is_p1_larger_than_p2(1, 0))
        print('Is p2 larger than p3? ', Z.is_p1_larger_than_p2(1, 2))  
        print('Is p3 larger than p2? ', Z.is_p1_larger_than_p2(2, 1))  
        
if __name__ == '__main__':
    unittest.main()