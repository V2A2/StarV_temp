# ------------- test for reach function -------------
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


class TestSatLinReach(unittest.TestCase):
    """
        Tests SatLin reach function
    """

    def test_reach(self):

        V = np.array([[0, 0], [1, 0], [0, 1]])
        print("\n V ------------------------ \n ", V)
        print("\n V' ------------------------ \n ", V.transpose())

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
        print("\n I------------------------n ", I.__repr__())

        lb_zono = np.array([-0.5, -0.5])
        ub_zono = np.array([0.5, 0.5])

        B_zono = Box(lb_zono, ub_zono)
        I_zono = B_zono.toZono()

        S1 = SatLin.reach(I, 'exact-star')
        # exact reach set using star
        print("\n S1 size ------------------------ \n", len(S1))
        print("\n S1 ------------------------ \n", S1)
        S2 = SatLin.reach(I, 'approx-star')
        # over-approximate reach set using star
        print("\n S2 ------------------------ \n", S2)
        S3 = SatLin.reach(I_zono, 'approx-zono')
        # over-approximate reach set using Zonotope
        print("\n S3 ------------------------ \n", S3)


if __name__ == '__main__':
    unittest.main()

# ------------- end the test for reach function -------------
