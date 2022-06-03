import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoConvexHullWithLinearTransform(unittest.TestCase):
    """
        Tests getting bounds of Zono with clip method
    """
    
    def test_convexHull_with_linearTransform(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        c = np.array([1, 1])
        V = np.random.rand(2, 3)
        Z1 = Zono(c, V)
        
        W = np.array([[2, 1], [0, -1]])
        Z2 = Z1.affineMap(W)
        
        Z12 = Z1.convexHull(Z2)
        print("Zonotope with convexHull\n")
        print(Z12.__repr__())
        print(Z12.__str__()) 

        print("Zonotope with convexHull_with_linearTransform\n")
        Z3 = Z1.convexHull_with_linearTransform(W)
        print(Z3.__repr__())
        print(Z3.__str__()) 
    
if __name__ == '__main__':
    unittest.main()