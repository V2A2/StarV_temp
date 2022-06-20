# ------------- test for Layer Construction function -------------
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


class TestLayerConstructor(unittest.TestCase):
    """
        Tests Layer constructor
    """

    def test_constructor(self):

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

        W = np.array([[1.5, 1], [0, 0.5]])
        print("\n W ------------------------ \n ", W)

        b = np.array([0.5, 0.5])
        print("\n b ------------------------ \n ", b)

        L = Layer(W, b, 'poslin')
        print("\n L poslin ------------------------ \n", L.__repr__())

        L1 = Layer(W, b, 'satlin')
        print("\n L satlin ------------------------ \n", L1.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for Layer Construction function -------------
