# ------------- test for stepReachStarApprox function -------------
import unittest
import copy
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/set/zono/")
sys.path.insert(0, "engine/set/box/")

from box import Box
from zono import Zono
from star import Star
from poslin import PosLin


class TestPosLinStepReachStarApprox(unittest.TestCase):
    """
        Tests PosLin stepReachStarApprox function
    """

    def test_stepReachStarApprox(self):

        V = np.array([[0, 1, 1], [0, 1, 0]])
        print("\n V ------------------------ \n ", V)

        C = np.array([
            [-0.315003530434092, -0.568434180229367],
            [0.770941699569057, -0.150899330325962],
            [0.672949951974609, 0.310136250794432],
            [-0.471143803780317, 0.528332704254980],
            [-0.556337350487487, 0.500351152064443],
        ])
        print("\n C ------------------------ \n ", C)

        d = np.array([
            0.760036419233332, 0.618771595964738, 0.671530988175932,
            0.706320090167365, 0.663428577226175
        ])
        print("\n d ------------------------ \n ", d)

        lb = np.array([-1.59838537713372, -1.60748823421369])
        ub = np.array([0.860829800317711, 1.57817118125545])
        print("\n lb ------------------------ \n ", lb)
        print("\n ub ------------------------ \n ", ub)

        # ------------- other V, C, d, lb and ub examples -------------
        # V = np.array([[0, 0.2500, 0.5000], [0, 0.7500, -1]])
        # C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        # d = np.array([1, 1, 1, 1])
        # lb = np.array([-1, -1])
        # ub = np.array([1, 1])

        I = Star(V, C, d, lb, ub)
        print("\n I ------------------------ \n", I.__repr__())

        S = PosLin.stepReachStarApprox(I, 0)
        print("\n S ------------------------ \n", S.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for stepReachStarApprox function -------------
