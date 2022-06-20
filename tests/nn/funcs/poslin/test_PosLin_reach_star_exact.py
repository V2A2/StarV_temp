# ------------- test for reach_star_exact function -------------
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


class TestPosLinReachStarExact(unittest.TestCase):
    """
        Tests PosLin reach_star_exact function
    """

    def test_reach_star_exact(self):

        lb = np.array([-0.5, -0.5])
        ub = np.array([0.5, 0.5])
        print("\n lb ------------------------ \n", lb)
        print("\n ub ------------------------ \n", ub)

        B = Box(lb, ub)
        print("\n B ------------------------ \n", B)

        I = Box.toZono(B)
        print("\n I ------------------------n", I)

        A = np.array([[0.5, 1], [1.5, -2]])
        print("\n A ------------------------ \n ", A)

        b = np.array([])
        I = I.affineMap(A, b)
        print("\n I ------------------------ \n ", I.__repr__())

        I1 = Zono.toStar(I)
        print("\n I1 ------------------------ \n", I1.__repr__())

        S1 = PosLin.reach_star_exact(I1, b)
        print("\n S1 ------------------------ \n", S1.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- end of the test for reach_star_exact function -------------

# ------------- Unused Testing -------------

# lb = np.matrix('-0.5; -0.5')
# ub = np.matrix('0.5; 0.5')

# A = np.matrix('0.5, 1; 1.5, -2')
# b = np.matrix([])

# flatten_ub = np.ndarray.flatten(ub, "F")
# map = np.argwhere(flatten_ub <= 0)
# ub_map = np.array([])
# for i in range(len(map)):
#     index = map[i][1] * len(flatten_ub[0]) + map[0][i]
#     ub_map = np.append(ub_map, index)
# print("\n ub_map ------------\n", ub_map)

# S2 = np.column_stack([I1])
# print(S2)
# print(len(S2[0]))
# S2 = np.column_stack([I1, I1])
# print(len(S2[0]))

# ------------- End of Unused Testing -------------
