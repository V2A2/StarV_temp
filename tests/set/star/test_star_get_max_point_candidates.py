import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarGetMaxPointCandidates(unittest.TestCase):
    """
        Tests get_max_point_candidates() function that estimates quickly max-point candidates
    """

    def test_concatenate(self):
        """
            Generate a star set and perform get_max_point_candidates()

            Output -> Star                
        """
        V = np.array([[0, 1, 0], [0, 0, 1]])
        C = np.array([[0.99985, 0.01706], [-0.40967, 0.91224], [-0.57369, -0.81907]])
        d = np.array([0.95441, 0.46047, -0.82643])
        predicate_lb = np.array([0.43863, 0.34452])
        predicate_ub = np.array([0.94867, 0.92634])
        S = Star(V, C, d, predicate_lb, predicate_ub)
        print('\nPrint S in detail: \n')
        print(S.__repr__())
        
        max_cands = S.get_max_point_candidates()
        print('max-point candidates: %s' % max_cands)
    
if __name__ == '__main__':
    unittest.main()