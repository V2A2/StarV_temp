import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarIsEmptySet(unittest.TestCase):
    """
        Tests to check if the Star set is an empty set, which refers it is an infeasible set.
    """

    def test_isEmptySet(self):
        """
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                True -> star is an empty set
                False -> star is a feasible set
                else -> error code from Gurobi LP solver
        """
        lb = np.array([1, 1])
        ub = np.array([2, 2])

        S = Star(lb, ub)
        print(S.__repr__())
        print(S.__str__())

        print("Is Star an empty set?", S.isEmptySet())
    
if __name__ == '__main__':
    unittest.main()