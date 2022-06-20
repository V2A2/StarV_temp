# ------------- test for reach_star_exact function -------------
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


class TestSatLinReachStarExact(unittest.TestCase):
    """
        Tests SatLin reach_star_exact function
    """

    def test_reach_star_exact(self):

        # ------------- Testing case 1 -------------

        # V = np.array([[0, 0], [1, 0], [0, 1]])
        # print("\n V ------------------------ \n ", V)
        # print("\n V' ------------------------ \n ", V.transpose())

        # A = np.array([[-0.0611698477587589, -0.770849850954382],
        #               [-0.0313891794166137, -0.770987945943988],
        #               [0.618164917579579, 0.119349313088699],
        #               [0.439711041323967, 0.692186698595730],
        #               [0.0508217184777092, 0.619924874435863],
        #               [-0.513873220787411, -0.192831817535829]])
        # print("\n A ------------------------ \n ", A)

        # b = np.array([
        #     0.634073147995385, 0.636075708249122, 0.776934924005279,
        #     0.572303917883616, 0.783013603321585, 0.835912796351032
        # ])
        # print("\n b ------------------------ \n ", b)

        # # ------------- other lb and ub examples -------------
        # # lb = np.array([-0.5, -0.5])
        # # ub = np.array([0.5, 0.5])
        # # lb = np.array([-1, -1])
        # # ub = np.array([1, 1])

        # lb = np.array([-2.1673, -0.8831])
        # ub = np.array([1.4273, 1.4408])
        # print("\n b ------------------------ \n ", lb)
        # print("\n ub ------------------------ \n ", ub)

        # I = Star(V.transpose(), A, b, lb, ub)
        # print("\n I-----------\n ", I.__repr__())

        # # ------------- End of Testing case 1 -------------

        # # ------------- Testing case 2 -------------

        Ai = np.array([[-0.540814703979925, -0.421878816995180],
                       [0.403580749757606, -0.291562729475043],
                       [0.222355769690372, 0.164981737653923],
                       [-0.391349781319239, 0.444337590813175],
                       [-0.683641719399254, -0.324718758259433]])
        print("\n Ai ------------------------ \n ", Ai)

        bi = np.array([
            0.727693424272787, 0.867244921118684, 0.960905270006411,
            0.805859450556812, 0.653599057168295
        ])
        print("\n bi ------------------------ \n ", bi)

        Vi = np.array([[-1.28142280110204, 0.685008254671879],
                       [3.22068720143861, 1.48359989341389],
                       [0.468690315965779, -2.32571060511741],
                       [0.349675922629359, -1.27663092336119],
                       [1.79972069619285, 3.39872156367377]])
        print("\n Vi ------------------------ \n ", Vi)

        V = np.array([[0, 0], [1, 0], [0, 1]])
        print("\n V ------------------------ \n ", V)

        # V, C, d
        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])

        I = Star(V.transpose(), Ai, bi, lb, ub)
        print("\n I ------------------------ \n", I.__repr__())

        # # ------------- End of Testing case 2 -------------

        S1 = SatLin.reach_star_exact(I, '')
        print("\n S1 size ------------------------ \n", len(S1))
        print("\n S1 ------------------------ \n", S1.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- end of the test for reach_star_exact function -------------
