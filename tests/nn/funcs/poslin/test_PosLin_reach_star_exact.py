
# ----------------- test for reach_star_exact function ---------
from engine.nn.funcs.poslin import PosLin
import copy
import numpy as np
from engine.set.star import Star
from engine.set.zono import Zono
from engine.set.box import Box

lb = np.matrix('-0.5; -0.5')
ub = np.matrix('0.5; 0.5')
B = Box(lb, ub)
print('\nB----------------\n', B)

I = Box.toZono(B)
print('\nI ---------\n', I)

A = np.matrix('0.5, 1; 1.5, -2')
b = np.matrix([])
I = I.affineMap(A, b)
I1 = Zono.toStar(I)
print('\nI1 ---------\n', I1)
# S1 = np.column_stack([I1])
# S2 = np.column_stack

[lb, ub] = Star.estimateRanges(I1)
print('\nlb -------- \n', lb)
print('\nub -------- \n', ub)

S1 = PosLin.reach_star_exact(I1, b)
print('\nS1 ---------- \n', S1.__repr__())

flatten_ub = np.ndarray.flatten(ub, 'F')
map = np.argwhere(flatten_ub <= 0)
ub_map = np.array([])
for i in range(len(map)):
    index = map[i][1] * len(flatten_ub[0]) + map[0][i]
    ub_map = np.append(ub_map, index)
print('\n ub_map ------------\n', ub_map)

S2 = np.column_stack([I1])
print(S2)
print(len(S2[0]))
S2 = np.column_stack([I1, I1])
print(len(S2[0]))

# ----------------- end of the test for reach_star_exact function