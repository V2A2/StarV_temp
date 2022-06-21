import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarConcatenate(unittest.TestCase):
    """
        Tests concatenate() function that concatanates the current Star with other Star, X
    """

    def test_concatenate(self):
        """
            Generate two star sets, S1 and S2, and concatenate S2 into S1 

            Output -> Star                
        """
        V1 = np.array([[0, 1, 0], [0, 0, 1]])
        C1 = np.array([[0.99985, 0.01706], [-0.40967, 0.91224], [-0.57369, -0.81907]])
        d1 = np.array([0.95441, 0.46047, -0.82643])
        predicate_lb_1 = np.array([0.43863, 0.34452])
        predicate_ub_1 = np.array([0.94867, 0.92634])
        S1 = Star(V1, C1, d1, predicate_lb_1, predicate_ub_1)
        print('\nPrint S1 in detail: \n')
        print(S1.__repr__())
        
        V2 = np.array([[0, 1, 0], [0, 0, 1]])
        C2 = np.array([[0.06731, 0.99773], [0.66075, -0.75061], [-0.99147, -0.13036]])
        d2 = np.array([ 0.92446, -0.08122, -0.27468])
        predicate_lb_2 = np.array([0.15661, 0.31556])
        predicate_ub_2 = np.array([0.86347, 0.916])
        S2 = Star(V2, C2, d2, predicate_lb_2, predicate_ub_2)
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