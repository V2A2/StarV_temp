# ------------- test for stepReach function -------------
import unittest
import sys
import numpy as np
import copy

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/set/zono/")
sys.path.insert(0, "engine/set/box/")

from poslin import PosLin
from zono import Zono
from star import Star


class TestPosLinStepReach(unittest.TestCase):
    """
        Tests PosLin stepReach function
    """

    def test_stepReach(self):

        # ------------- other lb and ub examples -------------
        # lb = np.array([-0.5, -0.5])
        # ub = np.array([0.5, 0.5])

        lb = np.array([-1, -1])
        ub = np.array([1, 1])
        print("\n lb ------------------------ \n ", lb)
        print("\n ub ------------------------ \n ", ub)

        I = Star(lb, ub)
        print("\n I------------------------ \n ", I.__repr__())

        W = np.array([[2, 1], [1, -1]])
        print("\n W------------------------ \n ", W)

        I = I.affineMap(W, np.array([]))
        print("\n I Affine ------------------------ \n", I)

        S = PosLin.stepReach(I, 0)
        print("\n S ------------------------ \n", S)


if __name__ == '__main__':
    unittest.main()

# ------------- end of the test for stepReach function -------------

# ------------- Unused Testing -------------
# index = 0
# c = copy.deepcopy(I.V[index, 0])
# print("\nc -----------\n", c)

# V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
# print("\nV ---------------- \n", V)

# new_C = np.vstack([I.C, V])
# print('\nnew_C -----------\n', new_C)

# new_d = np.hstack([I.d, -c])
# print('\nnew_d -----------\n', new_d)
# print('\nnew_d -----------\n', len(new_d.shape))

# new_V = copy.deepcopy(I.V)
# new_V[index, :] = np.zeros([1, I.nVar + 1])

# c1 = copy.deepcopy(I.Z.c)
# c1[index] = 0
# print('\nc1 ----------\n', c1)

# V1 = copy.deepcopy(I.Z.V)
# V1[index, :] = 0
# print('\nV1 -----------\n', V1)

# new_Z = Zono(c1, V1)
# print('\nnew_Z -----------\n', new_Z, new_Z.c, new_Z.V, new_Z.dim)

# S1 = Star(new_V, new_C, new_d, I.predicate_lb, I.predicate_ub, new_Z)
# S1.Z = new_Z
# print('S1 -----------\n', S1)

# new_C1 = np.vstack([I.C, -V])
# print('\nnew_C1 ------------\n', new_C1)

# new_d1 = np.hstack([I.d, c])
# print('\nnew_d ------------\n', new_d)

# S2 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub, I.Z)
# S2.Z = I.Z
# print('S2 -----------\n', S2)

# S = np.column_stack([S1, S2])
# S = []
# S.append(S1)
# S.append(S2)
# print('\nS ----------- \n', S)

# x1 = I.V[index, 0] + I.V[index, 1:I.nVar + 1] * I.predicate_lb
# print('\nx1 --------------\n', x1)
# x2 = I.V[index, 0] + I.V[index, 1:I.nVar + 1] * I.predicate_ub
# print('\nx2 --------------\n', x2)

# c = copy.deepcopy(I.V[index, 0])
# print('\nc -------------\n', c)

# V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
# print('\nV -------------\n', V)

# new_C = np.vstack([I.C, V])
# print('\nnew_C -------------\n', new_C)

# new_d = np.vstack([I.d, -c])
# print('\nnew_d -------------\n', new_d)

# new_V = copy.deepcopy(I.V)
# print('\nnew_V -------------\n', new_V)

# new_V[index, :] = np.zeros([1, I.nVar + 1])
# print('\nnew_V -------------\n', new_V)

# S = PosLin.stepReach2(I, 0)
# print("\nS----------\n", S)

# ------------- End of Unused Testing -------------