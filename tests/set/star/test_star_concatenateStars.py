import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConcatenateStars(unittest.TestCase):
    """
        Tests concatenateStars() function that concatanates many stars into a signle star
    """

    def test_isEmptySet(self):
        """
            Randomely generate an 1D numpy array of two star sets and concatenate them into a single star 

            Output -> Star                
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S1 = Star.rand(dim, N)
        print('\nPrint S1 in detail: \n')
        print(S1.__repr__())
        
        S2 = Star.rand(dim, N)
        print('\nPrint S2 in detail: \n')
        print(S2.__repr__())
        
        S3 = S1.concatenate(S2)
        print('\nPrint S3 in detail: \n')
        print(S3.__repr__())
        
        S4 = Star.concatenateStars(np.array([S1, S2]))
        print('\nPrint S4 in detail: \n')
        print(S4.__repr__())

    
if __name__ == '__main__':
    unittest.main()