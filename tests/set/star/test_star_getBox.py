import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarGetBox(unittest.TestCase):
    """
        Tests finding a box bound of a star set
    """
    
    def test_getBox(self):
        """
            Tests with initializing Star based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
            
            Output :
                Box ->
                    dim -> dimension of a Box
                    lb -> lower bound vector (1D numpy array)
                    ub -> upper bound vector (1D numpy array)
        """
        lb = np.array([49, 25, 9, 20])
        ub = np.array([51, 25.2, 11, 20.2])
        
        B1 = Box(lb, ub)
        print(B1.__repr__())
        
        S = B1.toStar()
        
        B2 = S.getBox()
        print(B2.__repr__())
        
if __name__ == '__main__':
    unittest.main()