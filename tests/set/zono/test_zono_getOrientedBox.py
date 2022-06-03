import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono")
from zono import *

class TestZonoGetOrientedBox(unittest.TestCase):
    """
        Tests getOrientedBox() function that an oriented rectangular hull enclosing the zonotope
    """
    
    def test_getOrientedBox(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        c1 = np.array([0, 0])
        V1 = np.array([[1, -1], [1, 1], [0.5, 0], [-1, 0.5]]) 
        Z1 = Zono(c1, V1.transpose())
        print('Z1: ', Z1.__repr__())
        
        B1 = Z1.getOrientedBox()
        print('B1: ', B1.__repr__())
            
if __name__ == '__main__':
    unittest.main()