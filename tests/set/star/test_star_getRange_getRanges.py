import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarGetRangeGetRanges(unittest.TestCase):
    """
        Tests getRange() function that finds lower bound vector and 
        upper bound vector of the state variables.
        
        Tests getRanges() function finds lower bound vector and 
        upper bound vector of the state variables
    """
    def test_getRange_and_getRanges(self):
        dim = 5
        lb = -2*np.ones(dim)
        ub = 2*np.ones(dim)
        
        S = Star(lb, ub)

        W = np.random.rand(dim,dim)
        b = np.random.rand(dim)
        S1 = S.affineMap(W, b)
        
        mins = np.empty(dim)
        maxs = np.empty(dim)
        print('getRange:\n')
        for i in range(dim):
            [mins[i], maxs[i]] = S1.getRange(i)
        print('min:', mins)
        print('max:', maxs)

        print('\ngetRanges:')
        [xmin, xmax] = S1.getRanges()
        print('min:', xmin)
        print('max:', xmax)

        if (mins != xmin).all():
            print('(min) getRange() is not equivalent to getRanges()')
        if (maxs != xmax).all():
            print('(max) getRange() is not equivalent to getRanges()')

if __name__ == '__main__':
    unittest.main()