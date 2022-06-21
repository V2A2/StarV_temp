import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConcatenateWithVector(unittest.TestCase):
    """
        Tests concatenate() function that concatanates the current Star with a vector, v (1D numpy array)
    """

    def test_concatenate_with_vector(self):
        """
            Randomely generate a Star set and concatenate it with a vector

            Output -> Star                
        """
        # dimension of a star
        dim = 2
        # number of star constraints
        N = 5

        S = Star.rand(dim, N)
        print('\nRandomely generated star S: \n')
        print(S.__repr__())
        
        v = np.random.rand(4)
        print('\n Randomely generated vector v: \n')
        print(v)
        
        Sv = S.concatenate_with_vector(v)
        print('\nConcatenated Star with a vector, Sv: \n')
        print(Sv.__repr__())
    
if __name__ == '__main__':
    unittest.main()