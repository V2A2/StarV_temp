# # -----------------test for reach function -------------------
from engine.nn.funcs.poslin import PosLin
import copy
import numpy as np
from engine.set.star import Star
from engine.set.zono import Zono
from engine.set.box import Box
lb = np.matrix('-0.5; -0.5')
ub = np.matrix('0.5; 0.5')

B = Box(lb=lb, ub=ub)
I = B.toZono()

A = np.matrix('0.5, 1; 1.5, -2')
b = np.matrix([])
I = I.affineMap(A, b)

I1 = I.toStar()
S1 = PosLin.reach(I1, 'exact-star') # exact reach set using star
print("S1 ---------: \n", S1)
S2 = PosLin.reach(I1, 'approx-star') # over-approximate reach set using star
print("S2 ---------: \n", S2)
# # -----------------end the test for reach function ------------------