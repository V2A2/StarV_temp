import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoGetRanges(unittest.TestCase):
    """
        Tests getting ranges of Zono
    """
    
    def test_getRanges(self):
        """
            Tests the initialization of Zono with:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                np.array([
                    lb : float -> lower bound (1D numpy array)
                    ub : float -> upper bound (1D numpy array)
                ])
        """
        c = np.array([0, 0])
        V = np.array([[1, -1], [1, 1]])
        Z = Zono(c, V)
        print("Initial zonotope")
        print(Z.__repr__())
        
        W = np.random.rand(3, 2)
        b = np.random.rand(3)
        
        Za = Z.affineMap(W, b)
        print("Affine mapped zonotope")
        print(Za.__repr__())
        
        [lb, ub] = Za.getBounds()
        print('Bounds lb: \n', lb)
        print('Boudns ub: \n', ub)
        
        [lb, ub] = Za.getRanges()
        print('Ranges lb: \n', lb)
        print('Ranges ub: \n', ub)        
    
if __name__ == '__main__':
    unittest.main()