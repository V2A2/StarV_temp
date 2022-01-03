#!/usr/bin/python3
import sys
import os
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.rstar import RStar
from engine.nn.funcs.tansig import TanSig

def main():
    np.set_printoptions(precision=25)
    dim = 2
    lb = -np.ones((2, 1))
    ub = np.ones((2, 1))

    S1 = Star(lb = lb, ub = ub)
    R1 = RStar(lb = lb, ub = ub)
    S2 = Star(lb = lb, ub = ub)
    R2 = RStar(lb = lb, ub = ub)

    W1 = 2*np.matrix(np.random.rand(dim,dim)) - 1 
    b1 = 2*np.matrix(np.random.rand(dim,1)) - 1
    W2 = 2*np.matrix(np.random.rand(dim,dim)) - 1 
    b2 = 2*np.matrix(np.random.rand(dim,1)) - 1

    S1 = S1.affineMap(W1, b1)
    R1 = R1.affineMap(W1, b1)
    S2 = S2.affineMap(W2, b2)
    R2 = R2.affineMap(W2, b2)
    print('\n-----after affine mapping-----')
    [s1_lb, s1_ub] = S1.getRanges()
    [r1_lb, r1_ub] = R1.getRanges()
    print('1st star lb: \n', s1_lb)
    print('1st star ub: \n', s1_ub)
    print('1st rstar lb: \n', r1_lb)
    print('1st rstar ub: \n', r1_ub)
    [s2_lb, s2_ub] = S2.getRanges()
    [r2_lb, r2_ub] = R2.getRanges()
    print('2nd star lb: \n', s2_lb)
    print('2nd star ub: \n', s2_ub)
    print('2nd rstar lb: \n', r2_lb)
    print('2nd rstar ub: \n', r2_ub)

    S1 = TanSig.reach_absdom_approx(S1)
    R1 = TanSig.reach_rstar_approx(R1)
    S2 = TanSig.reach_absdom_approx(S2)
    R2 = TanSig.reach_rstar_approx(R2)
    print('\n-----after reach_star_approx-----')
    [s1_lb, s1_ub] = S1.getRanges()
    [r1_lb, r1_ub] = R1.getRanges()
    print('1st star lb: \n', s1_lb)
    print('1st star ub: \n', s1_ub)
    print('1st rstar lb: \n', r1_lb)
    print('1st rstar ub: \n', r1_ub)
    [s2_lb, s2_ub] = S2.getRanges()
    [r2_lb, r2_ub] = R2.getRanges()
    print('2nd star lb: \n', s2_lb)
    print('2nd star ub: \n', s2_ub)
    print('2nd rstar lb: \n', r2_lb)
    print('2nd rstar ub: \n', r2_ub)

    S3 = S1.Sum(S2)
    R3 = R1.Sum(R2)
    print('\n-----after Minkowski Sum-----')
    [s3_lb, s3_ub] = S3.getRanges()
    [r3_lb, r3_ub] = R3.getRanges()
    [r3_lb_exact, r3_ub_exact] = R3.getExactRanges()
    print('3rd star lb: \n', s3_lb)
    print('3rd star ub: \n', s3_ub)
    print('3rd rstar lb: \n', r3_lb)
    print('3rd rstar ub: \n', r3_ub)
    print('3rd rstar lb exact: \n', r3_lb_exact)
    print('3rd rstar ub exact: \n', r3_ub_exact)
    print(S3.__repr__())
    print(R3.__repr__())

if __name__ == '__main__':
    main()   