import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarResetRow(unittest.TestCase):
    """
        Tests resetRow() function that resets a row of a star to zero
    """

    def test_resetRow(self):
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
        S_reset = S.resetRow(map)
        print('\nPrint Star after resetRow() in detail: \n')
        print(S_reset.__repr__())

    
if __name__ == '__main__':
    unittest.main()