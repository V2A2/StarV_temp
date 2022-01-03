#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.zono import Zono

def main():
    c1 = np.matrix('0; 0')
    V1 = np.matrix('1 -1; 1 1')
    Z1 = Zono(c1, V1)
    B1 = Z1.getBox()

    W = np.matrix('3 1; 1 0; 2 1')
    b = np.matrix('0.5; 1; 0')
    Z2 = Z1.affineMap(W, b)

    [lb, ub] = Z2.getBounds()
    print('lb: \n', lb)
    print('ub: \n', ub)

    
if __name__ == '__main__':
    main()
