import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarGetMinGetMax(unittest.TestCase):
    """
        Tests getMin() function and getMax() function that
        find lower bound and upper bound of state variable 
        at specific position using LP solver
    """
    
    def test_getMin_and_getMax(self):
        """
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                    
            Output:
                > getMin():
                    xmin -> lower bound
                    
                > getMax():
                    xmax -> upper bound
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
        print('getMin and getMax:\n')
        for i in range(dim):
            mins[i] = S1.getMin(i)
            maxs[i] = S1.getMax(i)
        print('min:', mins)
        print('max:', maxs)

        print('\ngetRanges:')
        [xmin, xmax] = S1.getRanges()
        print('min:', xmin)
        print('max:', xmax)

        if (mins != xmin).all():
            print('(min) getMin() is not equivalent to getRanges()')
        if (maxs != xmax).all():
            print('(max) getMax() is not equivalent to getRanges()')

if __name__ == '__main__':
    unittest.main()