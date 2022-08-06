# ------------- test for half_space_contains function -------------
import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/halfspace/")
from halfspace import HalfSpace


class TestHalfSpcaeContain(unittest.TestCase):
    """
        Tests for half_space_contains function
    """

    def test_Contain(self):

        G = np.array([[1, 0, 0], [1, -1, 1]])
        g = np.array([1, 2])
        U = HalfSpace(G, g)
        print("\n U ------------------------ \n", U.__repr__())

        x = np.array([[2], [1], [0]])
        bool_value = U.contains(x)
        print("\n bool_value ------------------------ \n", bool_value)


if __name__ == '__main__':
    unittest.main()

# ------------- end the test for half_space_contains function -------------