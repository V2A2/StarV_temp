import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box")
from box import *

class TestBoxConstructor(unittest.TestCase):
    """
        Tests Box constructor
    """
    
    def test_bounds_init(self):
        """
            Tests the initialization of Box with:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
            
            Output :
                Box ->
                    dim -> dimension of a Box
                    lb -> lower bound vector (1D numpy array)
                    ub -> upper bound vector (1D numpy array)
        """
        dim = 2
        lb = -np.ones(dim)
        ub = np.ones(dim)

        B = Box(lb, ub)
        print(B.__repr__())
        print(B.__str__())
        B.plot()


if __name__ == '__main__':
    unittest.main()
