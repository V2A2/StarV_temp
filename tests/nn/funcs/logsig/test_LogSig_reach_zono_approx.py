#!/usr/bin/python3

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box
from engine.set.zono import Zono
from engine.nn.funcs.logsig import *

def main():
    
    lb = np.matrix('-1;1')
    ub = np.matrix('1;2')

    I = Box(lb, ub)
    I = I.toZono()

    W = np.matrix('0.5 1; -1 1')
    I = I.affineMap(W)

    Z = LogSig.reach_zono_approx(I)
    print(Z.__repr__)
    [lb, ub] = Z.getRanges()
    print('lb: \n', lb)
    print('ub: \n', ub)
    
if __name__ == '__main__':
    main()