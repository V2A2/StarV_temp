# -------------test for reach function -------------
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


class TestPosLinReach(unittest.TestCase):
    """
        Tests PosLin reach function
    """

    def test_reach(self):

        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])
        print("\n lb ------------------------ \n ", lb)
        print("\n ub ------------------------ \n ", ub)

        B = Box(lb, ub)
        print("\n B ------------------------ \n ", B)
        I = B.toZono()
        print("\n I ------------------------ \n ", I)

        A = np.array([[0.5, 1], [1.5, -2]])
        print("\n A ------------------------ \n ", A)
        b = np.array([])
        I = I.affineMap(A, b)
        print("\n I ------------------------ \n ", I)

        I1 = I.toStar()
        print("\n I1 ------------------------ \n ", I1)

        I1_toList = []
        I1_toList.append(I1)
        S1 = PosLin.reach(I1_toList, 'exact-star')
        # exact reach set using star
        print("\n S1 ------------------------ \n", S1)
        S2 = PosLin.reach(I1, 'approx-star')
        # over-approximate reach set using star
        print("\n S2 ------------------------ \n", S2)
        S3 = PosLin.reach(I1, 'approx-star2')
        # over-approximate method 2 reach set using star
        print("\n S3 ------------------------ \n", S3)
        S4 = PosLin.reach(I, 'approx-zono')
        # over-approximate reach set using star
        print("\n S4 ------------------------ \n", S4)


if __name__ == '__main__':
    unittest.main()

# ------------- end the test for reach function -------------

# ------------- Unused Testing -------------

# lb = np.matrix('-0.5; -0.5')
# ub = np.matrix('0.5; 0.5')
# A = np.matrix('0.5, 1; 1.5, -2')
# b = np.matrix([])

# ------------- End of Unused Testing -------------