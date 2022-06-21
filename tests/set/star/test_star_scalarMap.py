import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarRand(unittest.TestCase):
    """
        Tests scalarMap() function that does scalar map of a Star:
            S' = alp * S, 0 <= alp <= alp_max
    """

    def test_rand(self):
        """
            Randomely generate star set and perform scalar map
                
            Output -> Star                
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S = Star.rand(dim, N)
        print('\nPrint randomely generated Star in detail: \n')
        print(S.__repr__())
        
        S1 = S.scalarMap(0.7)
        print('\nPrint Star after scalarMap() in detail: \n')
        print(S1.__repr__())
    
if __name__ == '__main__':
    unittest.main()