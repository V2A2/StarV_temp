import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoGetVertices(unittest.TestCase):
    """
        Tests getVertices() function that gets all vertices of a zonotope
    """
    
    def test_getVertices(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                all vertices of the zonotope
        """
        c = np.array([1, 1])
        V = np.array([[2, 1, 1], [-1, 1, 0]])
        Z = Zono(c, V)

        V = Z.getVertices()
        print("V: ", V)

if __name__ == '__main__':
    unittest.main()