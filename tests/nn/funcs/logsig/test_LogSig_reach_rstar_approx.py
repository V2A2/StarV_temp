#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.rstar import RStar
from engine.nn.funcs.logsig import LogSig

def main():
    np.set_printoptions(precision=25)
    dim = 4
    lb = -np.ones((dim ,1))
    ub = np.ones((dim, 1))

    S = Star(lb = lb, ub = ub)
    RS = RStar(lb = lb, ub = ub)
    
    W = 2*np.matrix(np.random.rand(dim,dim)) - 1 
    b = 2*np.matrix(np.random.rand(dim,1)) - 1

    S = S.affineMap(W, b)
    RS = RS.affineMap(W, b)
    
    print('\nafter affine mapping\n')
    [s_lb, s_ub] = S.getRanges()
    [rs_lb, rs_ub] = RS.getRanges()
    print('star lb: \n', s_lb)
    print('star ub: \n', s_ub)
    print('rstar lb: \n', rs_lb)
    print('rstar ub: \n', rs_ub)

    print('after reach_star_approx')
    S = LogSig.reach_absdom_approx(S)
    RS = LogSig.reach_rstar_approx(RS)

    [s_lb, s_ub] = S.getRanges()
    [rs_lb, rs_ub] = RS.getRanges()
    print('star lb: \n', s_lb)
    print('star ub: \n', s_ub)
    print('rstar lb: \n', rs_lb)
    print('rstar ub: \n', rs_ub)
    
if __name__ == '__main__':
    main()