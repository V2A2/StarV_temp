#!/usr/bin/python3
import sys
import os
import numpy as np
import gurobipy as gp
from gurobipy import GRB

os.chdir('.')

from engine.set.box import *
from engine.set.imagezono import ImageZono
from engine.set.zono import *
from engine.set.star import *
# from imagestar import *


# sys.path.insert(0, '/engine/nn/set')

# from engine.box_en import Box
# from engine.set.nn.star import Star

lb = np.matrix('-1; -1')
ub = np.matrix('1; 1')

print('lb: ', lb)
print('ub: ', ub)

print("\n----------------Box----------------")
B = Box(lb,ub)
print(B)
print("\naffineMap----------------Box")
W = np.matrix('0.5 0.5; 0.5 -0.5')
b = np.matrix('1; 0')
B = B.affineMap(W, b)
print(B)
print("\n")

print("\n----------------Zono----------------")
B = Box(lb,ub)
Z = B.toZono()
print(Z)
print("\naffineMap----------------Zono")
Z = Z.affineMap(W, b)
print(Z)

print("\n----------------Star----------------Z")
S = Z.toStar()
print(S)
print("\n----------------Star----------------B")
S = B.toStar()
print(S)
print("W: \n", W)
print("b: \n", b)
print("\naffineMap----------------Star")
S = S.affineMap(W, b)
print(S)
print("isEmptySet: ", S.isEmptySet())
print("Range(0): ", S.getRange(0))
print("Range(1): ", S.getRange(1))
print("Ranges: ", S.getRanges())
[lb, ub] = S.getRanges()
print("Ranges lb: ", lb)
print("Ranges ub: ", ub)
print("getBox: ", S.getBox())

# # Create a new model
# # m = gp.Model("matrix1")
# m = gp.Model()

# # Create variables
# x = m.addMVar (shape=2, name ="x", )

# # Set objective
# # obj = np.array([1.0, 1.0, 2.0])
# # obj = np.array([-1, -1/3])
# obj = -np.array([1, 1/3])
# print("obj: ", obj)
# print(obj.dtype)
# m.setObjective(obj @ x, GRB.MINIMIZE)

# # Build (sparse) constraint matrix
# val = np.array([1.0, 2.0, 3.0, -1.0, -1.0])
# row = np.array([0, 0, 0, 1, 1])
# col = np.array([0, 1, 2, 0, 1])

# # A = sp.csr_matrix((val, (row, col)), shape=(2, 3))
# M = np.matrix('1 1; 1 0.25; 1 -1; -0.25 -1; -1 -1; -1 1')
# A = sp.csr_matrix(M)
# print("A: ", A.toarray())
# print(A.dtype)

# # Build rhs vector
# # rhs = np.array([4.0, -1.0])
# rhs = np.array([2., 1., 2., 1., -1., 2.])
# print("rhs: ", rhs)
# print(rhs.dtype)
# # Add constraints
# m.addConstr(A @ x <= rhs, name="c")

# # Optimize model
# m.optimize()

# print(x.X)
# print('Obj: %g' % m.objVal)


# print("----------------check___gurobi---------------1")
# print("x: ", x)
# # print("obj: ", obj)
# print("val: ", val)
# print("row: ", row)
# print("col: ", col)
# print("A: ", A)
# print("A.array: ", A.toarray())
# print("rhs: ", rhs)

# print("----------------check___gurobi---------------2")
# A = np.matrix('1 0; 0 1')
# C = sp.csr_matrix(A)
# print("C: ", C)
# print("C_array: ", C.toarray())
# m2 = gp.Model()
# A = sp.csr_matrix(W)
# x = m2.addMVar(shape=2)
# print("x: ", x)
# # b = np.array([1.0, 1.0])
# b = np.array([4.0, -1.0])
# print("b: ", b)
# m2.addConstr(A @ x <= b)
# m2.optimize()

print("\n----------------imageZono----------------1")
#  attack on pixel (1,1) and (1,2)
L1 = np.matrix('-0.1 -0.2 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0') 
L2 = np.matrix('-0.1 -0.15 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
L3 = L2
LB = np.array([L1, L2, L3])
print("LB: ", LB)
print("shape LB: ", LB.shape)
U1 = np.matrix('0 0.2 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
U2 = np.matrix('0.1 0.15 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
# U1 = np.matrix('0 0.2 0; 0 0 0; 0 0 0; 0 0 0')
# U2 = np.matrix('0.1 0 0.15; 0 0 0; 0 0 0; 0 0 0')
U3 = U2
UB = np.array([U1, U2, U3])

print("UB: \n", UB)
print("shape UB: ", UB.shape)
print("shape U1: ", U1.shape)
print("len of shape UB: ", len(UB.shape))
print("len of shape U1: ", len(U1.shape))
print("len UB: ", len(UB))
print("len U1: ", len(U1))
print("size comp: ", UB.shape == LB.shape)

UB_flat = UB.flatten()
print("flatten: ", UB_flat)
print("UB reshape: \n", UB_flat.reshape(3,4,4))

print("\n----------------imageZono----------------2")
print("LB: \m", LB)
print("UB: \n", UB)
image_Z = ImageZono(lb_image = LB, ub_image = UB)

# stack_UB = np.array(UB, LB)
# print("stack_UB: ", stack_UB)
# print("stack_UB.shape: ", stack_UB.shape)

# S = B.toStar()
# print(S)
# print("W: \n", W)
# print("b: \n", b)
print("\naffineMap----------------imageZono")