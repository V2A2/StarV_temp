import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoGetBounds(unittest.TestCase):
    """
        Tests getting bounds of Zono with clip method
    """
    
    def test_getBounds(self):
        """
            Tests with initializing Zono (zonotope) based on
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                np.array([
                    lb -> lower bound (1D numpy array)
                    ub -> upper bound (1D numpy array)
                ])
        """
        c = np.array([0, 0])
        V = np.array([[1, -1], [1, 1]])
        Z = Zono(c, V)
        print("test zonotope")
        print(Z.__repr__())
        
        W = np.array([[3, 1], [1, 0], [2, 1]])
        b = np.array([0.5, 1, 0])
        
        Za = Z.affineMap(W, b)
        print("Affine mapped zonotope\n")
        print(Za.__repr__())
        
        [lb, ub] = Za.getBounds()
        print('Bounds lb: \n', lb)
        print('Boudns ub: \n', ub)
        
        [lb, ub] = Za.getRanges()
        print('Ranges lb: \n', lb)
        print('Ranges ub: \n', ub)        

    
if __name__ == '__main__':
    unittest.main()