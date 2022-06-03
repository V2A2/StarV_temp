import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxAffineMap(unittest.TestCase):
    """
        Tests affine mapping of Box
    """
    
    def test_affineMap(self):
        """
            Test affine mapping -> W * Box + b

            W : affine map scale
            b : affine map offset
            
            Output :
                Box ->
                    dim -> dimension of a Box
                    lb -> lower bound vector (1D numpy array)
                    ub -> upper bound vector (1D numpy array)
        """
        
        dim = 4
        lb = -np.ones(dim)
        ub = np.ones(dim)
        
        B = Box(lb, ub)
        
        W = np.array([[0.5, 1.0, -0.4, -0.6], [0.0, -0.2, 0.8, 0.3], [0.1, -0.2, 0.4, -0.1], [0.9, 0.0, 0.1, 0.2]])
        b = np.array([0.4, -0.3, -0.7, 0.1])
        print('W:\n%s\n' % W)
        print('b:\n%s\n' % b)
        
        am_result = B.affineMap(W, b)
        print(am_result.__repr__())
        

if __name__ == '__main__':
    unittest.main()