# ------------- test for reach_zono_approx function -------------
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


class TestPosLinReachZonoApprox(unittest.TestCase):
    """
        Tests PosLin reach_zono_approx function
    """

    def test_reach_zono_approx(self):

        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])
        print("\n lb ------------------------ \n", lb)
        print("\n ub ------------------------ \n", ub)

        B = Box(lb, ub)
        print("\n B ------------------------ \n", B.__repr__())
        I1 = B.toZono()
        print("\n I1 ------------------------ \n", I1.__repr__())

        A = np.array([[0.5, 1], [1.5, -2]])
        b = np.array([])
        I = I1.affineMap(A, b)
        print("\n I ------------------------ \n", I.__repr__())

        Z = PosLin.reach_zono_approx(I)
        print("\n Z ------------------------ \n", Z.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End of the test for reach_zono_approx function -------------
