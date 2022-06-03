import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxToZono(unittest.TestCase):
    """
        Tests the conversion from Box to Zono set
    """
    
    def test_toZono(self):
        """
            Tests with initializing Zono (zonotope) based on
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
            
            Output :
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        dim = 3
        lb = np.ones(dim)
        ub = -np.ones(dim)
        
        B = Box(lb, ub)
        result_zono = B.toZono()
        print(result_zono.__repr__())
        print(result_zono.__str__())


if __name__ == '__main__':
    unittest.main()