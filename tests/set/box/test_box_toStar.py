import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxToStar(unittest.TestCase):
    """
        Tests the conversion from Box to Star set
    """

    def test_toStar(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                Star ->
                    V -> Basis matrix (2D numpy array)
                    C -> Predicate matrix (2D numpy array)
                    d -> Predicate vector (1D numpy array)
                    predicate_lb -> predicate lower bound vector (1D numpy array)
                    predicate_ub -> predicate upper bound vector (1D numpy array)
        """
        input_dim = 3
        lb = np.ones(input_dim)
        ub = -np.ones(input_dim)

        print('\n-----------------------input box------------------------')
        B = Box(lb, ub)
        print(B.__repr__())

        S = B.toStar()
        print('\n---------------------box toStar()-----------------------')
        print(S.__repr__())

        # affine mapping
        W = np.random.rand(input_dim, input_dim)
        b = np.random.rand(input_dim)
        
        print('\n-------------------affine mapped box--------------------')
        Ba = B.affineMap(W, b)
        print(Ba.__repr__())
        print('\n---------------affine mapped box toStar()---------------')
        Sb = B.toStar()
        print(Sb.__repr__())
        print('\n-------------------affine mapped star-------------------')
        Sba = Sb.affineMap(W, b)
        print(Sba.__repr__())

    
if __name__ == '__main__':
    unittest.main()