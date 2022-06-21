import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarMinkowskiSum(unittest.TestCase):
    """
        Tests MinkowskiSum() function that computes Minkowski sum of two stars
    """

    def test_convexHull(self):
        """
            Randomely generate 2 Stars and compute Minkowski sum
            
            Output -> Star                
        """
        # -1 <= a[1] <= 1, -1 <= a[2] <= 2
        V1 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        C1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
        d1 = np.array([1, 1, 1, 1])
        
        S1 = Star(V1, C1, d1)
        print('\nPrint S1: \n')
        print(S1.__repr__())
        
        W = np.array([[2, 1, 1], [1, 0, 2], [0, 1, 0]])
        b = np.array([0.5, 0.5, 0])
        print('\nPrint W: \n')
        print(W)
        print('\nPrint b: \n')
        print(b)
        
        V2 = np.array([[1, 0], [0, 1], [1, 1]])
        C2 = np.array([[1], [-1]])
        d2 = np.array([0.5, 0.5])
        
        S2 = Star(V2, C2, d2)
        print('\nPrint S2: \n')
        print(S2.__repr__())
        
        S3 = S1.affineMap(W, b)
        print('\nPrint S3: \n')
        print(S3.__repr__())
        
        S12 = S1.MinkowskiSum(S2)
        S13 = S1.MinkowskiSum(S3)

        print('\nPrint S12: \n')
        print(S12.__repr__())
        
        print('\nPrint S13: \n')
        print(S13.__repr__())
    
if __name__ == '__main__':
    unittest.main()