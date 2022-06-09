import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoGetSupInfinityNorm(unittest.TestCase):
    """
        Tests getSupInfinityNorm() function
    """
    
    def test_getSupInfinityNorm(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                sup infinity norm
        """
        c1 = np.array([0, 0])
        V1 = np.array([[1, -1], [1, 1]])
        Z1 = Zono(c1, V1)
        r1 = Z1.getSupInfinityNorm()
        
        c2 = np.array([1, 1])
        V2 = np.array([[2, 1], [-1, 1]])
        Z2 = Zono(c2, V2)
        r2 = Z2.getSupInfinityNorm()
        
        print("r1: ", r1)
        print("\nr2: ", r2)

if __name__ == '__main__':
    unittest.main()