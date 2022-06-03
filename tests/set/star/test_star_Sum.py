import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
from star import *

class TestStarSum(unittest.TestCase):
    """
        Tests Minkowski Sum of a star
    """
    def test_Sum(self):
        """
            Tests with initializing Star based on:
                Star :
                    V : basis matrix (2D numpy array)
                    C : predicate matrix (2D numpy array)
                    d : predicate vector (1D numpy array)
        """
        np.set_printoptions(precision=25)
        V1 = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
        C1 = np.array([[1, 0], [-1, 0] ,[0, 1], [0, -1]])
        d1 = np.array([1, 1, 1, 1])
        S1 = Star(V1, C1, d1)
        
        V2 = np.array([[1, 0], [0, 1], [1, 1]])
        C2 = np.array([[1],[-1]])
        d2 = np.array([0.5, 0.5])
        S2 = Star(V2, C2, d2)

        W = np.array([[2, 1, 1], [1, 0, 2], [0, 1, 0]])
        b = np.array([0.5, 0.5, 0])

        S3 = S1.affineMap(W, b)
        
        S12 = S1.Sum(S2)
        S13 = S1.Sum(S3)

        print('\nS1:\n', S1.__repr__())
        print('\nS2:\n', S2.__repr__())
        print('\nS3:\n', S3.__repr__())

        print('\nS12:\n', S12.__repr__())
        print('\nS13:\n', S13.__repr__())

if __name__ == '__main__':
    unittest.main()