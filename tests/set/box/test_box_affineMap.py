#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

def main():
    dim = 4
    lb = -np.ones((dim, 1))
    ub = np.ones((dim, 1))

    B = Box(lb,ub)
    print(B.__repr__())

    # W = np.random.rand(dim, dim)
    # b = np.random.rand(dim,1)

    W = np.matrix('0.5, 1.0, -0.4, -0.6; 0.0, -0.2, 0.8, 0.3; 0.1, -0.2, 0.4, -0.1; 0.9, 0.0, 0.1, 0.2')
    b = np.matrix('0.4; -0.3; -0.7; 0.1')

    print('W:\n%s\n' % W)
    print('b:\n%s\n' % b)

    B = B.affineMap(W,b)
    print(B.__repr__())
    

if __name__ == '__main__':
    main()