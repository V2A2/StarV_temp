import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConstructor(unittest.TestCase):
    """
        Tests Star constructor
    """
    
    def test_basic_init(self):
        """
            Tests the initialization with:
                V : Basis matrix (2D numpy array)
                C : Predicate matrix (2D numpy array)
                d : Predicate vector (1D numpy array)
                predicate_lb : predicate lower bound (1D numpy array)
                predicate_ub : predicate upper bound (1D numpy array)
                
            Output:
                Star ->
                    V -> Basis matrix (2D numpy array)
                    C -> Predicate matrix (2D numpy array)
                    d -> Predicate vector (1D numpy array)
                    predicate_lb -> predicate lower bound (1D numpy array)
                    predicate_ub -> predicate upper bound (1D numpy array)
        """
        c = np.zeros([3, 1])
        I = np.eye(3)
        # The basis matrix is 2D numpy array
        # V is 3 x 4 matrix
        V = np.hstack([c, I])
        # The predicate matrix is 2D numpy array
        # C is 10 x 3 matrix
        C = np.array([[1.4665, -0.2385, 0.7890],
                    [ 0.3087,  0.2045, -1.3436],
                    [ 0.2307, -0.0285,  0.1064],
                    [-0.4090,  1.2930, -0.2055],
                    [-0.5055,  1.4825,  0.3096],
                    [ 0.5621,  0.8597,  1.4414],
                    [ 0.0040, -1.2719,  0.8599],
                    [ 0.4446, -0.0730, -1.6252],
                    [-0.4807, -0.7488, -0.5396],
                    [-0.8750, -0.4378, -1.1899]])
        # The predicate vector is 1D numpy array
        # d is a vector (array) with 10 variables
        d = np.ones(10)
        # The predicate lower bound and upper bound vectors are 1D numpy arrays
        # predicate_lb and predicate_ub are vectors with 3 variables
        predicate_lb = np.array([-2.1534, -1.1242, -0.7466])
        predicate_ub = np.array([ 1.2214,  1.0513,  1.1760])
        S = Star(V, C, d, predicate_lb, predicate_ub)
        print('\nPrint all information of star in detail:\n')
        print(S.__repr__())
        print('\n\nPrint inormation of star in short: \n')
        print(S.__str__())
        
        print('\n Is Stary an empty set? %s' % S.isEmptySet())
        
    def test_bounds_init(self):
        """
            Tests the initialization with:
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
        
        completion_flag = True
        
        # dimension of Star
        dim = 2
        # Star can be implemented by lower and upper bound vectors, e.g. L-infinity norm attack.
        lb = -2*np.ones(dim)
        ub = 2*np.ones(dim)
        
        S = Star(lb, ub)
        print('\nPrint all information of star in detail: \n')
        print(S.__repr__())
        print('\n\nPrint inormation of star in short: \n')
        print(S.__str__())
    
if __name__ == '__main__':
    unittest.main()