#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star

def main():
    dim = 5
    lb = -2*np.ones((dim, 1))
    ub = 2*np.ones((dim, 1))

    S = Star(lb = lb, ub = ub)

    W = np.random.rand(dim, dim)
    b = np.random.rand(dim, 1)

    S1 = S.affineMap(W, b)
    
    mins = np.empty((dim, 1))
    maxs = np.empty((dim, 1))
    print('estimateBound:\n')
    for i in range(dim):
        [mins[i], maxs[i]] = S1.estimateBound(i)
    print('min:', mins)
    print('max:', maxs)

    print('\nestimateBounds:')
    [xmin, xmax] = S1.estimateBounds()
    print('min:', xmin)
    print('max:', xmax)

    if (mins != xmin).all():
        print('(min) estimateBound() is not equivalent to estimateBounds()')
    if (maxs != xmax).all():
        print('(max) estimateBound() is not equivalent to estimateBounds()')


if __name__ == '__main__':
    main()   