import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarScaleRow(unittest.TestCase):
    """
        Tests scaleRow() function that scales a row of a star
    """

    def test_scaleRow(self):
        """
            Randomely generate star set by providing 
            dim : dimension of a star to be generated
            N : maximum number of constraints to be generated on a star
                
            Output -> Star                
        """
        # dimension of a star
        dim = 3
        # number of star constraints
        N = 5

        S = Star.rand(dim, N)
        print('\nPrint randomely generated Star in detail: \n')
        print(S.__repr__())
        
        map = np.array([0, 1])
        scale_value = 2
        S_scaled = S.scaleRow(map, scale_value)
        print('\nPrint Star after scaleRow() in detail: \n')
        print(S_scaled.__repr__())
            
if __name__ == '__main__':
    unittest.main()