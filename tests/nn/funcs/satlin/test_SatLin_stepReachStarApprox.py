# ------------- test for stepReachStarApprox function -------------
import unittest
import copy
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/satlin/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/set/zono/")
sys.path.insert(0, "engine/set/box/")

from box import Box
from zono import Zono
from star import Star
from satlin import SatLin


class TestSatLinStepReachStarApprox(unittest.TestCase):
    """
        Tests SatLin stepReachStarApprox function
    """

    def test_stepReachStarApprox(self):

        V = np.array([[0, 0], [1, 0], [0, 1]])
        print("\n V ------------------------ \n ", V)
        print("\n V------------------------ \n ", V.transpose())

        A = np.array([[-0.0611698477587589, -0.770849850954382],
                      [-0.0313891794166137, -0.770987945943988],
                      [0.618164917579579, 0.119349313088699],
                      [0.439711041323967, 0.692186698595730],
                      [0.0508217184777092, 0.619924874435863],
                      [-0.513873220787411, -0.192831817535829]])
        print("\n A ------------------------ \n ", A)

        b = np.array([
            0.634073147995385, 0.636075708249122, 0.776934924005279,
            0.572303917883616, 0.783013603321585, 0.835912796351032
        ])
        print("\n b ------------------------ \n ", b)

        # ------------- other lb and ub examples -------------
        # lb = np.array([-0.5, -0.5])
        # ub = np.array([0.5, 0.5])
        # lb = np.array([-1, -1])
        # ub = np.array([1, 1])

        lb = np.array([-2.1673, -0.8831])
        ub = np.array([1.4273, 1.4408])
        print("\n lb ------------------------ \n ", lb)
        print("\n ub ------------------------ \n ", ub)

        I = Star(V.transpose(), A, b, lb, ub)
        print("\n I ------------------------ \n ", I.__repr__())

        S1 = SatLin.stepReachStarApprox(I, 1, 'gurobi')
        print("\n S1 ------------------------ \n", S1.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- end of the test for stepReachStarApprox function -------------
