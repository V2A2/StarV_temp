import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoMinkowskiSum(unittest.TestCase):
    """
        Tests Minkowski Sum of a Zono
    """
    
    def test_MinkowskiSum(self):
        """
            Tests with initializing Zono based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        c1 = np.array([0, 0])
        V1 = np.array([[1, -1], [1, 1]])
        Z1 = Zono(c1, V1)
        
        c2 = np.array([1, 1])
        V2 = np.array([[2, 1], [-1, 1]])
        Z2 = Zono(c2, V2)
        
        result_zono = Z1.MinkowskiSum(Z2)
        print(result_zono.__repr__())
        print(result_zono.__str__()) 

    
if __name__ == '__main__':
    unittest.main()