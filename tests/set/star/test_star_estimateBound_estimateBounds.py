import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarEstimateBoundEstimateBounds(unittest.TestCase):
    """
        Tests estimageBound() function and estimateBounds() function that 
        estimate lower bound and upper bound vector of state variable
    """
    
    def test_estimateBound_and_estimateBounds(self):
        """
            estimateBound(): 
                Estimates lower bound and upper bound of state variable at specific index using clip method from Stanely Bak.
            
            estimateBounds():
                Estimates lower bound vector and upper bound vector of state variable
            
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                > estimateBound():
                    np.array([
                        lb -> lower bound
                        ub -> upper bound
                    ])
                > estimateBounds():
                    np.array([
                        lb -> lower bound vector (1D numpy array)
                        ub -> upper bound vector (1D numpy array)
                    ])
        """
        dim = 5
        lb = -2*np.ones(dim)
        ub = 2*np.ones(dim)

        S = Star(lb, ub)

        W = np.random.rand(dim, dim)
        b = np.random.rand(dim)

        S1 = S.affineMap(W, b)
        
        mins = np.empty(dim)
        maxs = np.empty(dim)
        print('estimateBound:\n')
        for i in range(dim):
            [mins[i], maxs[i]] = S1.estimateBound(i)
        print('min:', mins)
        print('max:', maxs)

        print('\nestimateBounds:')
        [xmin, xmax] = S1.estimateBounds()
        print('min:', xmin)
        print('max:', xmax)

        if (mins != xmin).all():
            print('(min) estimateBound() is not equivalent to estimateBounds()')
        if (maxs != xmax).all():
            print('(max) estimateBound() is not equivalent to estimateBounds()')

if __name__ == '__main__':
    unittest.main()