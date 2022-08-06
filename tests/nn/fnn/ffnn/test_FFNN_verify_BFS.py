# ------------- test for FFNN_verify_F_DFS_SingleCore function -------------
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


class TestFFNNVerifyDFSSingleCore(unittest.TestCase):
    """
        Tests FFNN_verify_F_DFS_SingleCore function
    """

    def test_FFNN_verify_DFS(self):

        W1 = np.array([[1, -1],[0.5, 2],[-1, 1]])
        print("\n W1 ------------------------ \n", W1)

        b1 = np.array([-1, 0.5, 0])
        print("\n b1 ------------------------ \n", b1)

        W2 = np.array([[-2, 1, 1],[0.5, 1, 1]])
        print("\n W2 ------------------------ \n", W2)

        b2 = np.array([-0.5, -0.5])
        print("\n b2 ------------------------ \n", b2)

        L1 = Layer(W1, b1, 'poslin') # construct first layer
        print("\n L1 poslin ------------------------ \n", L1.__repr__())

        L2 = Layer(W2, b2, 'poslin') # construct second layer
        print("\n L2 satlin ------------------------ \n", L2.__repr__())

        lb = np.array([-2, -1]) # lower-bound vector of input set
        print("\n lb ------------------------ \n", lb)

        ub = np.array([2, 2]) # upper-bound vector of input set
        print("\n ub ------------------------ \n", ub)

        I = Star(lb, ub) # construct input set
        print("\n I ------------------------ \n", I.__repr__())

        G = np.array([[-1, 0]])
        g = np.array([-5])
        U = HalfSpace(G, g)
        print("\n U ------------------------ \n", U.__repr__())

        F = FFNN([L1, L2])
        print("\n F ------------------------ \n", F.__repr__())

        [safe, CEx] = F.verify_BFS_SingleCore('InputSet', I, 'UnsafeRegion', U)
        print("\n safe ------------------------ \n", safe)
        print("\n CEx ------------------------ \n", CEx.__repr__())

        # [safe1, CEx1] = F.verify_BFS_SingleCore('InputSet', I, 'UnsafeRegion', U, 'ReachMethod', 'approx-star')
        # print("\n safe1 ------------------------ \n", safe1)
        # print("\n CEx1 ------------------------ \n", CEx1.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for FFNN_verify_F_DFS_SingleCore function -------------
