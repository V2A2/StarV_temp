import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestGetPredicateBounds(unittest.TestCase):
    """
        Tests getPredicateBounds() function gets bounds of predicate variables
    """

    def test_getPredicateBounds(self):
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

        S1 = Star.rand(dim, N)
        print('\nRandomely generated star: \n')
        print(S1.__repr__())
        print("\nPredicate bounds: %s \n" % (S1.getPredicateBounds()))
        
        S2 = Star(S1.V, S1.C, S1.d) 
        print('\nRandomely generated star: \n')
        print(S2.__repr__())
        print("\nPredicate bounds: %s \n" % (S2.getPredicateBounds()))

    
if __name__ == '__main__':
    unittest.main()