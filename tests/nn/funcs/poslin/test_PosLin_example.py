# import copy
#
# import numpy as np
#
# #test for step reach func in poslin
# from engine.nn.funcs.poslin import PosLin
# from engine.set.star import Star
#
# a = np.matrix('1, 2, 3; 4, 5, 6; 7, 8, 9')
# b = np.matrix('1, 2, 3; 4, 5, 6; 7, 8, 9')
# c = np.vstack([a, b])
# print('\nc ---------\n', c)
# d = np.row_stack([a, b])
# print('\nd ---------\n', d)
#
# from engine.set.zono import Zono
# if not isinstance(I.Z, Zono):
#     print('dgfhjdgkfd')
#
# a = np.matrix([[1,2],[3,4]])
# b = np.matrix([[5,6],[7,8]])
# print(np.matrix.flatten(a))
#
# print(map1)
# 61
# V1 = I.V;
# print(V1)
# V1[0, :] = 0
# print(V1)
#
# 76
# c = I.V[0, 0]
# V = I.V[0, 1:I.nVar + 1]
# new_C = np.vstack([I.C, V])
# new_d = np.vstack([I.d, -c])
# new_V = I.V
# new_V[0, :] = np.zeros((1, I.nVar + 1))
# new_V[0, :] = 0
# print(new_V)
#
# 121
# x1 = I.V[0, 0] + I.V[0,1:I.nVar + 1] * I.predicate_lb
# x2 = I.V[0,0] + I.V[0,1:I.nVar + 1] * I.predicate_ub
# print(x2)
#
# c = I.V[0, 0]
# V = I.V[0, 1:I.nVar + 1]
# new_C = np.vstack([I.C, V])
# new_d = np.vstack([I.d, -c])
# new_V = I.V
# new_V[0, :] = np.zeros([1, I.nVar + 1])
# print(new_V)
#
# 286
# [lb, ub] = Star.estimateRanges(I)
#
# print(lb)
# print(ub)
#
# flatten_ub = np.ndarray.flatten(ub, 'F')
# flatten_lb = np.ndarray.flatten(lb, 'F')
#
# print(len(flatten_lb[0]))
# # 1x2 [[3 2]]
# print(flatten_ub)
#
# #[0 0]
# #[0 1]
# map = np.argwhere(flatten_lb < 0 and flatten_ub > 0)
# print(map)
#
# index = map[1][1] * len(flatten_lb[0]) + map[1][0]
#
# new_map = np.array([])
# for i in range(len(map)):
#     index = map[i][1] * len(flatten_ub[0]) + map[0][i]
#     print(index)
#     new_map = np.append(new_map, index)
# print(new_map)
#
#
# lb = np.matrix('0.5 0.5 0.5; 0.5 0.5 0.5')
# #ub = np.matrix('0.1 0.2 0.5; 0.5 0.5 0.5')
# ub = np.matrix('1 2 3; -1 2 -2')
# flatten_ub = np.ndarray.flatten(ub, 'F')
# map = np.argwhere(flatten_ub > 0)
# ub_map = np.array([])
# print(map)
# print(flatten_ub)
# for i in range(len(map)):
#     index = map[i][1]
#     print(index)
#     ub_map = np.append(ub_map, index)
# print(ub_map)