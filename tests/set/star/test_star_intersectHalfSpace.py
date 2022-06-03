import unittest

import sys
import numpy as np
import polytope as pc

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarIntersectHalfSpace(unittest.TestCase):
    """
        Tests intersection of a Star with a half space: 
        H(x) := Hx <= g
    """
    
    def test_intersectHalfSpace(self):
        """
             Tests with initializing Star based on:
                V : Basis matrix (2D numpy array)
                C : Predicate matrix (2D numpy array)
                d : Predicate vector (1D numpy array)
            
            H : halfspace matrix
            g : halfspace vector
            
            Output:
                Star ->
                    V -> Basis matrix (2D numpy array)
                    C -> Predicate matrix (2D numpy array)
                    d -> Predicate vector (1D numpy array)
        """
        # dimension of convex set
        dim = 3
        
        # number of constraints in polytope
        N = 4
        A = np.random.rand(N, dim)

        # compute the convex hull
        P = pc.qhull(A)

        V = np.array([[1, 1, 0, 1], [1, 0, 1, 1]])
        S = Star(V, P.A, P.b)
        print(S.__repr__())
        
        # Halfspace: x[1] >= 1
        H = np.array([[-1, 0]])
        g = np.array([-1])
        
        result_star = S.intersectHalfSpace(H, g)
        print(result_star.__repr__())
        
if __name__ == '__main__':
    unittest.main()