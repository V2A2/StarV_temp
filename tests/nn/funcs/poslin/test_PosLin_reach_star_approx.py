
# ----------------- test for reach_star_approx function ---------
from engine.nn.funcs.poslin import PosLin
import copy
import numpy as np
from engine.set.star import Star

V = np.matrix('0 1 1; 0 1 0')
C = np.matrix('-0.540814703979925 -0.421878816995180;'
              '0.403580749757606 -0.291562729475043;'
              '0.222355769690372 0.164981737653923;'
              '-0.391349781319239 0.444337590813175;'
              '-0.683641719399254 -0.324718758259433')
b = np.matrix('0.727693424272787;'
              '0.867244921118684;'
              '0.960905270006411;'
              '0.805859450556812;'
              '0.653599057168295')
lb = np.matrix('-1.28142280110204;'
               '-2.32571060511741')
ub = np.matrix('3.22068720143861;'
               '3.39872156367377')

V = np.matrix('0 0.2500 0.5000; 0 0.7500 -1')
C = np.matrix('1 0;'
              '0 1;'
              '-1 0;'
              '0 -1')
b = np.matrix('1;'
              '1;'
              '1;'
              '1')
lb = np.matrix('-1;'
               '-1')
ub = np.matrix('1;'
               '1')
I = Star(V=V, C=C, d=b, pred_lb=lb, pred_ub=ub)

lb = np.matrix('-1;-1')
ub = np.matrix('1;1')
from engine.set.star import Star
I = Star(lb=lb, ub=ub)
W = np.matrix('2 1; 1 -1')
I = I.affineMap(W, np.matrix([]))

print('\nI ---------- \n', I.__repr__())

#from glpk import glpk, GLPK

S = PosLin.reach_star_approx(I)
print('\nS ---------- \n', S.__repr__())

# ----------------- end of the test for reach_star_approx function ---------