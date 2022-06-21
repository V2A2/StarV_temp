import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConvexHull(unittest.TestCase):
    """
        Tests convexHull() function that computes convex hull of Stars
    """

    def test_convexHull(self):
        """
            Randomely generate 2 Stars and compute convex hull
            
            Output -> Star                
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S1 = Star.rand(dim, N)
        print('\nPrint S1 in detail: \n')
        print(S1.__repr__())
        
        S2 = Star.rand(dim, N)
        print('\nPrint S2 in detail: \n')
        print(S2.__repr__())

        S_convexHull = S1.convexHull(S2)
        print('\nPrint convex hull of two Stars: \n')
        print(S_convexHull.__repr__())
    
if __name__ == '__main__':
    unittest.main()