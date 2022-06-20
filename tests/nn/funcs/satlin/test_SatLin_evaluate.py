# ------------- test for evaluate function -------------
import unittest
import copy
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/satlin/")

from satlin import SatLin


class TestSatLinEvaluate(unittest.TestCase):
    """
        Tests SatLin Evaluate function
    """

    def test_evaluate(self):

        x = np.array([[-1], [0.5], [2]])
        print("\n x ------------------------: \n", x)

        y = np.array([-1, 0.5, 2])
        print("\n y ------------------------: \n", y)

        eva_x = SatLin.evaluate(x)
        print("\n eva_x ------------------------: \n", eva_x)

        eva_y = SatLin.evaluate(y)
        print("\n eva_y ------------------------: \n", eva_y)


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for evaluate function -------------
