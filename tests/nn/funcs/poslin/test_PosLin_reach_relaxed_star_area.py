# ------------- test for reach_relaxed_star_area function -------------
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


class TestPosLinReachRelaxedStarArea(unittest.TestCase):
    """
        Tests PosLin reach_relaxed_star_area function
    """

    def test_reach_star_approx(self):

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
        print("\n lb ------------------------ \n ", lb)

        ub = np.array([0.860829800317711, 1.57817118125545])
        print("\n ub ------------------------ \n ", ub)

        # ------------- other V, C, d, lb and ub examples -------------
        # V = np.array([[0, 0.2500, 0.5000], [0, 0.7500, -1]])
        # C = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
        # d = np.array([1, 1, 1, 1])
        # lb = np.array([-1, -1])
        # ub = np.array([1, 1])

        I = Star(V, C, d, lb, ub)
        print("\n I ------------------------ \n", I.__repr__())

        # from glpk import glpk, GLPK

        S = PosLin.reach_relaxed_star_area(I, 0.5, '', 'display')
        print("\n S ------------------------ \n", S.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- end of the test for reach_relaxed_star_area function -------------

# ------------- Unused Testing -------------
# V = np.matrix('0 1 1; 0 1 0')
# C = np.matrix('-0.540814703979925 -0.421878816995180;'
#               '0.403580749757606 -0.291562729475043;'
#               '0.222355769690372 0.164981737653923;'
#               '-0.391349781319239 0.444337590813175;'
#               '-0.683641719399254 -0.324718758259433')
# b = np.matrix('0.727693424272787;'
#               '0.867244921118684;'
#               '0.960905270006411;'
#               '0.805859450556812;'
#               '0.653599057168295')
# lb = np.matrix('-1.28142280110204;'
#                '-2.32571060511741')
# ub = np.matrix('3.22068720143861;'
#                '3.39872156367377')

# V = np.matrix('0 0.2500 0.5000; 0 0.7500 -1')
# C = np.matrix('1 0;'
#               '0 1;'
#               '-1 0;'
#               '0 -1')
# b = np.matrix('1;'
#               '1;'
#               '1;'
#               '1')
# lb = np.matrix('-1;'
#                '-1')
# ub = np.matrix('1;'
#                '1')
# ------------- End of Unused Testing -------------