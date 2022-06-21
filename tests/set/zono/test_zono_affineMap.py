import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono")
from zono import *

class TestZonoAffineMap(unittest.TestCase):
    """
        Tests affine mapping of Zono
    """
    
    def test_affineMap(self):
        """
            Test affine mapping -> W * Zono + b
        
            W : affine maping scale (weight matrix)
            b : affine maping offset (bias vector)
            
            Tests with initializing Zono (zonotope) based on
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
            
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        input_dim = 2
        c = np.zeros(input_dim)
        V = np.array([[1, -1], [1, 1]])
        
        Z = Zono(c, V)
        print('Initial zonotope\n')
        print(Z.__repr__())
        print(Z.__str__())
        
        W = np.random.rand(input_dim, 2)
        b = np.random.rand(2)
        print('Affine mapping matrix and bias vector\n')
        print('W: ', W)
        print('b: ', b)
        
        Za = Z.affineMap(W, b)
        print('Affine mapped zonotope\n')
        print(Za.__repr__())
        print(Za.__str__())
        Za.plot()

if __name__ == '__main__':
    unittest.main()