import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *


class TestZonoToBox(unittest.TestCase):
    """
        Tests conversion from Zono to Box set
    """
    
    def test_toBox(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output :
                Box ->
                    dim -> dimension of a Box
                    lb -> lower bound vector (1D numpy array)
                    ub -> upper bound vector (1D numpy array)
        """
        c = np.zeros(2)
        V = np.array([[1, -1], [1, 1]])
        Z = Zono(c, V)

        box_result = Z.getBox()
        
        print("Initial zonotope\n")
        print(Z.__repr__())
        print(Z.__str__())
        
        print("Converted box from zonotope\n")
        print(box_result.__repr__())
        print(box_result.__str__())

if __name__ == '__main__':
    unittest.main()
