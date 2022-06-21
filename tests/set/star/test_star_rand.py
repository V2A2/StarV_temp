import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarRand(unittest.TestCase):
    """
        Tests Star.rand() function that generates a random Star set
    """

    def test_rand(self):
        """
            Randomely generate star set by providing 
            dim : dimension of a star to be generated
            N : maximum number of constraints to be generated on a star
                
            Output -> Star                
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S = Star.rand(dim, N)
        print('\nPrint all information of star in detail: \n')
        print(S.__repr__())
        print('\n\nPrint inormation of star in short: \n')
        print(S.__str__())
        S.plot()
        plt.show()

        print("Is Star an empty set?", S.isEmptySet())
    
if __name__ == '__main__':
    unittest.main()