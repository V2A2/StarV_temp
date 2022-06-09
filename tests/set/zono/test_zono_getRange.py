import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoGetRange(unittest.TestCase):
    """
        Tests getting range of Zono
    """
    
    def test_getRanges(self):
        """
            Tests with initializing Zono (zonotope) based on
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                np.array([
                    lb -> lower bound of x[index] (1D numpy array)
                    ub -> upper bound of x[index] (1D numpy array)
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
        
        [lb, ub] = Za.getRanges()
        print('Ranges lb: \n', lb)
        print('Ranges ub: \n', ub) 
        
        for i in range(Za.dim):
            print("index: ", i)
            [l, u] = Za.getRange(i)
            print('Range lb: \n', l)
            print('Range ub: \n', u) 

    
if __name__ == '__main__':
    unittest.main()