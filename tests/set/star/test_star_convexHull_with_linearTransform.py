import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConvexHullWithLinearTransform(unittest.TestCase):
    """
        Tests convexHull_with_linearTransform() function that computes 
        convex hull of a Star with its linear transformation
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

        S = Star.rand(dim, N)
        print('\nPrint S in detail: \n')
        print(S.__repr__())
        
        L = np.array([[2, 1], [1, -1]])

        S_convexHull = S.convexHull_with_linearTransform(L)
        print('\nPrint convex hull of a Star with its linear transformation: \n')
        print(S_convexHull.__repr__())
    
if __name__ == '__main__':
    unittest.main()