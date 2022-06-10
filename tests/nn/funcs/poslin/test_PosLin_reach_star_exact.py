# ----------------- test for reach_star_exact function ---------
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

# lb = np.matrix('-0.5; -0.5')
# ub = np.matrix('0.5; 0.5')
lb = np.array([-0.5, -0.5])
ub = np.array([0.5, 0.5])
B = Box(lb, ub)
print("\nB----------------\n", B)

I = Box.toZono(B)
print("\nI ---------\n", I)

# A = np.matrix('0.5, 1; 1.5, -2')
# b = np.matrix([])
A = np.array([[0.5, 1], [1.5, -2]])
b = np.array([])
I = I.affineMap(A, b)
I1 = Zono.toStar(I)
print("\nI1 affine ---------\n", I1.__repr__())
# S1 = np.column_stack([I1])
# S2 = np.column_stack

# [lb, ub] = Star.estimateRanges(I1)
# print("\nlb -------- \n", lb)
# print("\nub -------- \n", ub)

S1 = PosLin.reach_star_exact(I1, b)
print("\nS1 ---------- \n", S1.__repr__())

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

# ----------------- end of the test for reach_star_exact function
