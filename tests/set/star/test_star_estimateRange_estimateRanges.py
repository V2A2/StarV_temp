#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star

def main():
    np.set_printoptions(precision=25)
    dim = 5
    lb = -2*np.ones((dim,1 ))
    ub = 2*np.ones((dim,1 ))

    S = Star(lb = lb, ub = ub)

    W = np.random.rand(dim, dim)
    b = np.random.rand(dim, 1)
    S1 = S.affineMap(W, b)

    mins = np.empty((dim, 1))
    maxs = np.empty((dim, 1))
    print('estimateRange:')
    for i in range(dim):
        [mins[i], maxs[i]] = S1.estimateRange(i)
    print('min:', mins)
    print('max:', maxs)

    print('\nestimateRanges:')
    [xmin, xmax] = S1.estimateRanges()
    print('min:', xmin)
    print('max:', xmax)

    print('\ngetRange:')
    [xmin_range, xmax_range] = S1.getRanges()
    print('min:', xmin_range)
    print('max:', xmax_range)

    if (mins != xmin).all():
        print('(min) estimateRange() is not equivalent to estimateRanges()')
    if (maxs != xmax).all():
        print('(max) estimateRange() is not equivalent to estimateRanges()')

if __name__ == '__main__':
    main()   