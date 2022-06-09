import unittest

import sys
import numpy as np
import polytope as pc

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarSample(unittest.TestCase):
    """
        Tests sampling number of points in the feasible Star set 
    """
    
    def test_sample(self):
        """
            N : number of points in the sample
            
            Output :
                V -> a set of at most N sampled points in the star set 
        """
        # dimension of polytope
        dim = 2
        # number of constraints in polytope
        N = 4
        A = np.random.rand(N, dim)

        # compute the convex hull
        P = pc.qhull(A)
        
        print('A: ', P.A)
        print('b: ', P.b)
        
        # convert polytope to star
        V = np.array([[0, 1, 0], [0, 0, 1]])
        S = Star(V, P.A, P.b)
        print(S.__repr__())

        X = S.sample(200)

        P.plot()
        for i in range(X.shape[1]):
            plt.plot(X[0, i], X[1, i], 'go')
        plt.show()

if __name__ == '__main__':
    unittest.main()