import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxGetVertices(unittest.TestCase):
    """
        Tests getVertices() function that gets all vertices of boxes
    """
    
    def test_getVertices(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                all vertices of the box
        """
        lb = np.array([-0.1, -0.2, -1])
        ub = np.array([1, 0.5, 1])
        B = Box(lb, ub)
        
        V = B.getVertices()
        print("V: ", V)
        
if __name__ == '__main__':
    unittest.main()