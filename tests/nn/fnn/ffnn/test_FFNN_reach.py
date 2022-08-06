# ------------- test for FFNN_reach function -------------
import unittest
import copy
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/nn/funcs/satlin/")
sys.path.insert(0, "engine/nn/layers/layer/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/nn/fnn/ffnn/")
sys.path.insert(0, "engine/set/halfspace/")

from halfspace import HalfSpace
from FFNN import FFNN
from star import Star
from poslin import PosLin
from satlin import SatLin
from layer import Layer


class TestFFNNreach(unittest.TestCase):
    """
        Tests FFNN_reach function
    """

    def test_FFNN_reach(self):

        W1 = np.array([[1, 1],[0, 1]])
        print("\n W1 ------------------------ \n", W1)

        b1 = np.array([0, 0.5])
        print("\n b1 ------------------------ \n", b1)

        L1 = Layer(W1, b1, 'poslin') # construct first layer
        print("\n L1 poslin ------------------------ \n", L1.__repr__())

        F = FFNN([L1])
        print("\n F ------------------------ \n", F.__repr__())

        lb = np.array([-1, -1]) # lower-bound vector of input set
        print("\n lb ------------------------ \n", lb)

        ub = np.array([1, 1]) # upper-bound vector of input set
        print("\n ub ------------------------ \n", ub)

        I = Star(lb, ub) # construct input set
        print("\n I ------------------------ \n", I.__repr__())

        [R1, t1] = F.reach(I, 'exact-star')
        print("\n t ------------------------ \n", t1)
        print("\n R ------------------------ \n", R1.__repr__())

        [R2, t2] = F.reach(I, 'approx-star2')
        print("\n t2 ------------------------ \n", t2)
        print("\n R2 ------------------------ \n", R2.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for FFNN_reach function -------------
