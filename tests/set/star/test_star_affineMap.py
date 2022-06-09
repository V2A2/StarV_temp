import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarAffineMap(unittest.TestCase):
    """
        Tests affine mapping of Star
    """
    
    def test_affineMap(self):
        """
            Test affine mapping -> W * Star + b
        
            W : affine maping scale (weight matrix)
            b : affine maping offset (bias vector)
            
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                Star ->
                    V -> basis matrix (2D numpy array)
                    C -> predicate matrix (2D numpy array)
                    d -> predicate vector (1D numpy array)
                    predicate_lb -> predicate lower bound vector (1D numpy array)
                    predicate_ub -> predicate upper bound vector (1D numpy array)
        """
        lb = np.array([1, 1])
        ub = np.array([2, 2])
    
        S = Star(lb, ub)
        print("Initial Star set\n")
        print(S.__repr__())
     
        W = np.array([[1, -1], [1, 1]])
        b = np.array([0.5, 0.5])
        
        am_S = S.affineMap(W, b)
        print("Affine mapped Star set\n")
        print(am_S.__repr__())    

if __name__ == '__main__':
    unittest.main()