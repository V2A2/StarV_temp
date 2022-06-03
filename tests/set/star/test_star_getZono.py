import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarGetZonot(unittest.TestCase):
    """
        Tests to find a zonotope bounding of a Star (an over-approximation of a Star using zonotope)
    """
    
    def test_getZono(self):
        """
            Tests with initializing Star based on
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output :
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        
        lb = np.array([-3,  -3])
        ub = np.array([2, 2])  

        S = Star(lb, ub)
        print('S: \n', S.__repr__())
        
        Z = S.getZono()
        print('\nZ: \n', Z.__repr__())
    
if __name__ == '__main__':
    unittest.main()  