import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *


class TestZonoToStar(unittest.TestCase):
    """
        Tests conversion from Zono to Star set
    """

    def test_toStar(self):
        """
            Tests with initializing Zono (zonotope) based on:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output :
                Star ->
                    V -> Basis matrix (2D numpy array)
                    C -> Predicate matrix (2D numpy array)
                    d -> Predicate vector (1D numpy array)
                    predicate_lb -> predicate lower bound (1D numpy array)
                    predicate_ub -> predicate upper bound (1D numpy array)
        """
        c = np.array([0, 0])
        V = np.array([[1, -1], [1, 1]])
        
        Z = Zono(c,V)
        print("Initial zonotope\n")
        print(Z.__repr__())
        print(Z.__str__())
    
        S = Z.toStar()
        print("Star converted from zonotope\n")
        print(S.__repr__())
        print(S.__str__())

if __name__ == '__main__':
    unittest.main()