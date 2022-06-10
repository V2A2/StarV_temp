# ---------------- test for stepReach function -------------------
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

# lb = np.array([-0.5, -0.5])
# ub = np.array([0.5, 0.5])
lb = np.array([-1, -1])
ub = np.array([1, 1])
# print("lb------------\n ",lb)
# print("ub------------\n ",ub)

I = Star(lb, ub)
# print("\nI-----------\n ",I.__repr__())

W = np.array([[2, 1], [1, -1]])
# print("\nW-----------\n ",W)

I = I.affineMap(W, np.array([]))
# print("\nI Affine ----------\n", I)
# print('I.v -----------\n', I.V)
# print('I.c -----------\n', I.C)
# print('I.d -----------\n', I.d)
# print('I.dim -----------\n', I.dim)
# print('I.nVar -----------\n', I.nVar)
# print('I.pre_lb -----------\n', I.predicate_lb)
# print('I.pre_ub -----------\n', I.predicate_ub)
# print('I.state_lb -----------\n', I.state_lb)
# print('I.state_ub -----------\n', I.state_ub)
# print('I.Z -----------\n', I.Z, I.Z.V, I.Z.c, I.Z.dim)

S = PosLin.stepReach(I, 0)
print("\nS----------\n", S)

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
# print('S1.v -----------\n', S1.V)
# print('S1.c -----------\n', S1.C)
# print('S1.d -----------\n', S1.d)
# print('S1.dim -----------\n', S1.dim)
# print('S1.nVar -----------\n', S1.nVar)
# print('S1.pre_lb -----------\n', S1.predicate_lb)
# print('S1.pre_ub -----------\n', S1.predicate_ub)
# print('S1.state_lb -----------\n', S1.state_lb)
# print('S1.state_ub -----------\n', S1.state_ub)
# print('S1.Z -----------\n', S1.Z, S1.Z.V, S1.Z.c, S1.Z.dim)

# new_C1 = np.vstack([I.C, -V])
# print('\nnew_C1 ------------\n', new_C1)

# new_d1 = np.hstack([I.d, c])
# print('\nnew_d ------------\n', new_d)

# S2 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub, I.Z)
# S2.Z = I.Z
# print('S2 -----------\n', S2)
# print('S2.v -----------\n', S2.V)
# print('S2.c -----------\n', S2.C)
# print('S2.d -----------\n', S2.d)
# print('S2.dim -----------\n', S2.dim)
# print('S2.nVar -----------\n', S2.nVar)
# print('S2.pre_lb -----------\n', S2.predicate_lb)
# print('S2.pre_ub -----------\n', S2.predicate_ub)
# print('S2.state_lb -----------\n', S2.state_lb)
# print('S2.state_ub -----------\n', S2.state_ub)
# print('S2.Z -----------\n', S2.Z)

# S = np.column_stack([S1, S2])
# S = []
# S.append(S1)
# S.append(S2)
# print('\nS ----------- \n', S)
# ---------------- end of the test for stepReach function -------------------

# ----------------- the test for stepReach2 function ------------------
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
# ----------------- end of the test for stepReach2 function ------------------
