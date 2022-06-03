import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxGetRanges(unittest.TestCase):
    """
        Tests get ranges of a Box
    """
    
    def test_getRanges(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                np.array([
                    lb -> lower bound (1D numpy array)
                    ub -> upper bound (1D numpy array)
                ])
        """
        
        dim = 4
        lb = -np.ones(dim)
        ub = np.ones(dim)
        
        B = Box(lb, ub)
        
        # affine mapping of Box
        W = np.array([[0.5, 1.0, -0.4, -0.6], [0.0, -0.2, 0.8, 0.3], [0.1, -0.2, 0.4, -0.1], [0.9, 0.0, 0.1, 0.2]])
        b = np.array([0.4, -0.3, -0.7, 0.1])
        
        Ba = B.affineMap(W, b)

        [lb, ub] = Ba.getRanges()
        print("lb: %s" % lb)
        print("ub: %s" % ub)
        
        print(Ba.__repr__())
        

if __name__ == '__main__':
    unittest.main()