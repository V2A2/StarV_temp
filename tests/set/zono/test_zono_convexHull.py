import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono")
from zono import *

class TestZonoConvexHull(unittest.TestCase):
    """
        Tests convex hull of zonotope with another zonotope
    """
    
    def test_convexHull(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """    
        c1 = np.array([0, 0])
        V1 = np.array([[1, 0, -1], [1, 1, 1]])
        Z1 = Zono(c1, V1)
        
        c2 = np.array([1, 1])
        V2 = np.array([[2, 1, 0], [-1, 1, 0]])
        Z2 = Zono(c2, V2)
        
        Z3 = Z2.convexHull(Z1)
        print(Z3.__repr__())
        print(Z3.__str__()) 
    
if __name__ == '__main__':
    unittest.main()