#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.nn.funcs.tansig import *

def main():
    
    lb = np.matrix('-0.1; -0.1')
    ub = np.matrix('0.1; 0.1')

    I = Star(lb = lb, ub = ub)
    # print(I.__repr__)
    [lb, ub] = I.getRanges()
    print('lb: \n', lb)
    print('ub: \n', ub)
    
    W = np.matrix('0.1 -1; 0.1 1')
    I = I.affineMap(W)
    print('\nafter affine mapping\n')
    # print(I.__repr__)
    [lb, ub] = I.getRanges()
    print('lb: \n', lb)
    print('ub: \n', ub)

    print('after reach_star_approx')
    S = TanSig.reach_absdom_approx(I)
    # print(S.__repr__)

    [lb, ub] = S.getRanges()
    print('lb: \n', lb)
    print('ub: \n', ub)
    
if __name__ == '__main__':
    main()