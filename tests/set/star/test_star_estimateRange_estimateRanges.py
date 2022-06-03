import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarEstimateRangeEstimateRanges(unittest.TestCase):
    """
        Tests estimageBound() function and estimateBounds() function that 
        estimate lower bound and upper bound vector of state variable
    """
    
    def test_estimateRange_and_estimateRanges(self):
        """
            estimateRange(): 
                Estimates range of a state variable at specific position
            
            estimateRanges():
                Estimates ranges of a state variable using clip method from Stanley Bak
            
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                > estimateRange():
                    np.array([
                        lb -> lower bound
                        ub -> upper bound
                    ])
                > estimateRanges():
                    np.array([
                        lb -> lower bound vector (1D numpy array)
                        ub -> upper bound vector (1D numpy array)
                    ])
        """
        np.set_printoptions(precision=25)
        dim = 5
        lb = -2*np.ones(dim)
        ub = 2*np.ones(dim)

        S = Star(lb, ub)

        W = np.random.rand(dim, dim)
        b = np.random.rand(dim)

        S1 = S.affineMap(W, b)

        mins = np.empty(dim)
        maxs = np.empty(dim)
        print('estimateRange:')
        for i in range(dim):
            [mins[i], maxs[i]] = S1.estimateRange(i)
        print('min:', mins)
        print('max:', maxs)

        print('\nestimateRanges:')
        [xmin, xmax] = S1.estimateRanges()
        print('min:', xmin)
        print('max:', xmax)

        print('\ngetRange:')
        [xmin_range, xmax_range] = S1.getRanges()
        print('min:', xmin_range)
        print('max:', xmax_range)

        if (mins != xmin).all():
            print('(min) estimateRange() is not equivalent to estimateRanges()')
        if (maxs != xmax).all():
            print('(max) estimateRange() is not equivalent to estimateRanges()')

if __name__ == '__main__':
    unittest.main()