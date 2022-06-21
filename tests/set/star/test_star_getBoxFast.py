import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarGetBoxFast(unittest.TestCase):
    """
        Tests getBoxFast() function that estimates ranges of the star vector quickly
        These ranges are not the exact ranges. 
        The function finds ranges that are over-approximation of the exact ranges
    """

    def test_getBoxFast(self):
        """
            Randomely generate star set by providing 
            dim : dimension of a star to be generated
            N : maximum number of constraints to be generated on a star
                
            Output : Box               
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S = Star.rand(dim, N)
        print('\nGenerated Star set: \n')
        print(S.__repr__())

        B1 = S.getBox()
        print('\nBox via getBox(): \n')
        print(B1.__repr__())
        
        B2 = S.getBoxFast()
        print('\nBox via getBoxFast(): \n')
        print(B2.__repr__())
    
if __name__ == '__main__':
    unittest.main()