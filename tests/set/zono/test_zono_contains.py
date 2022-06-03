import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono")
sys.path.insert(0, "engine/set/star")
from zono import *
from star import *

class TestZonoContains(unittest.TestCase):
    """
        Tests getOrientedBox() function to check if a zonotope contains a point
    """
    
    def test_contains(self):
        """
            Tests with initializing Zono converting from Star
                
            Output:
                True -> if the zonotope contain a point x
                False -> if the zonotope does not contain a point x
        """
        # dimension of polytope
        dim = 2
        # number of constraints in polytope
        N = 4
        A = np.random.rand(N, dim)

        # compute the convex hull
        P = pc.qhull(A)
        
        # convert polytope to star
        V = np.array([[0, 1, 0], [0, 0, 1]])
        S = Star(V, P.A, P.b)
        Z = S.getZono()

        X = S.sample(200)
        num_points = X.shape[1]
        print("num_points: ", num_points)
        
        x = 0
        for i in range(num_points-1):
            x += not Z.contains(X[:, i])

        # since zonotope is more conservative than star,
        # all points sampled by star should be inside zonotope
        print('Are there any points not in zonotope: ', x)
            
if __name__ == '__main__':
    unittest.main()