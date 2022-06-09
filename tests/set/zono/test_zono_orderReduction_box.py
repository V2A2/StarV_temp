import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoOrderReductionBox(unittest.TestCase):
    """
        Tests getting bounds of Zono with clip method
    """
    
    def test_orderReduction_box(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """    
        c = np.array([0, 0])
        V = np.array([[1, -1], [1, 1], [0.5, 1], [-1.2, 1]])
        Z = Zono(c, V.T)      
    
        result_zono = Z.orderReduction_box(3)
        print(result_zono.__repr__())
        print(result_zono.__str__()) 

    
if __name__ == '__main__':
    unittest.main()