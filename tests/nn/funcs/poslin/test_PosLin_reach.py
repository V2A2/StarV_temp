# # -----------------test for reach function -------------------
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
lb = np.array([-0.5, -0.5])
# ub = np.matrix('0.5; 0.5')
ub = np.array([0.5, 0.5])

B = Box(lb, ub)
I = B.toZono()

# A = np.matrix('0.5, 1; 1.5, -2')
# b = np.matrix([])
A = np.array([[0.5, 1], [1.5, -2]])
b = np.array([])
I = I.affineMap(A, b)

I1 = I.toStar()
S1 = PosLin.reach(I1, "exact-star")  # exact reach set using star
print("S1 ---------: \n", S1)
S2 = PosLin.reach(I1, "approx-star")  # over-approximate reach set using star
print("S2 ---------: \n", S2)
# # -----------------end the test for reach function ------------------
