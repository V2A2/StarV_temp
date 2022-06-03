import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarGetMinsGetMaxs(unittest.TestCase):
    """
        Tests getMins() and getMaxs() functions that
        find lower bound vector and upper bound vector of 
        state variable using LP solver
    """
    
    def test_getMin_and_getMaxs(self):
        """
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                    
            Output:
                > getMins():
                    np.array([
                        lb -> lower bound vector (1D numpy array)
                        ub -> upper bound vector (1D numpy array)
                    ])
                    
                > getMaxs():
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
        print('getMin and getMax:\n')
        for i in range(dim):
            mins[i] = S1.getMin(i)
            maxs[i] = S1.getMax(i)
        print('min:', mins)
        print('max:', maxs)

        print('\ngetMins and getMaxs:')
        map = np.arange(dim)
        xmin = S1.getMins(map)
        xmax = S1.getMaxs(map)
        print('min:', xmin)
        print('max:', xmax)

        print('\ngetMins parallel:')
        pxmin = S1.getMins(map, 'parallel')
        pxmax = S1.getMaxs(map, 'parallel')
        print('pmin:', pxmin)
        print('pmax:', pxmax)

        if (mins != xmin).all():
            print('(min) getMin() is not equivalent to getRanges()')
        if (maxs != xmax).all():
            print('(max) getMax() is not equivalent to getRanges()')
        if (pxmin != xmin).all():
            print('(min) getMins() single is not equivalent to parallel')
        if (pxmax != xmax).all():
            print('(max) getMaxs() single is not equivalent to parallel')

if __name__ == '__main__':
    unittest.main()