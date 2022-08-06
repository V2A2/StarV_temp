# ------------- test for Layer reach satlin function -------------
import unittest
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/nn/funcs/satlin/")
sys.path.insert(0, "engine/nn/layers/layer/")
sys.path.insert(0, "engine/set/star/")

from star import Star
from poslin import PosLin
from satlin import SatLin
from layer import Layer


class TestLayerReachSatLin(unittest.TestCase):
    """
        Tests Layer reach satlin function
    """

    def test_reach_satlin(self):

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

        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])

        I = Star(V.transpose(), Ai, bi, lb, ub)
        print("\n I ------------------------ \n", I.__repr__())

        W = np.array([[1.5, 1], [0, 0.5]])
        print("\n W ------------------------ \n ", W)

        b = np.array([0.5, 0.5])
        print("\n b ------------------------ \n ", b)

        L = Layer(W, b, 'satlin')
        print("\n L satlin ------------------------ \n", L.__repr__())

        I1 = I.affineMap(W, b)
        print("\n I1 ------------------------ \n", I1.__repr__())

        I_list = []
        I_list.append(I)
        #S = PosLin.reach(I1)
        S = L.reach(I_list, 'exact-star')
        print("\n S size ------------------------ \n", len(S))
        print("\n S ------------------------ \n", S.__repr__())
        S1 = L.reach(I_list, 'approx-star')
        print("\n S1 ------------------------ \n", S1.__repr__())

        # -------------  Old random input set -------------
        # Ai = np.matrix('-0.540814703979925 -0.421878816995180;'
        #               '0.403580749757606 -0.291562729475043;'
        #               '0.222355769690372 0.164981737653923;'
        #               '-0.391349781319239 0.444337590813175;'
        #               '-0.683641719399254 -0.324718758259433')

        # bi = np.matrix('0.727693424272787;'
        #               '0.867244921118684;'
        #               '0.960905270006411;'
        #               '0.805859450556812;'
        #               '0.653599057168295')

        # Vi = np.matrix('-1.28142280110204 0.685008254671879;'
        #               '3.22068720143861 1.48359989341389;'
        #               '0.468690315965779 -2.32571060511741;'
        #               '0.349675922629359 -1.27663092336119;'
        #               '1.79972069619285 3.39872156367377')

        # W = np.matrix('1.5 1;'
        #               '0 0.5')
        # b = np.matrix('0.5;'
        #               '0.5')
        # V = np.matrix('0 0;'
        #               '1 0;'
        #               '0 1')
        # ------------- End of Old random input set -------------


if __name__ == '__main__':
    unittest.main()

# ------------- end the test for Layer reach satlin function -------------
