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

    mins = np.empty((dim,1))
    maxs = np.empty((dim,1))
    print('getMin and getMax:\n')
    for i in range(dim):
        mins[i] = S1.getMin(i)
        maxs[i] = S1.getMax(i)
    print('min:', mins)
    print('max:', maxs)

    print('\ngetMins:')
    map = np.arange(dim)
    xmin = S1.getMins(map)
    xmax = S1.getMaxs(map)
    print('min:', xmin)
    print('max:', xmax)

    print('\ngetMins parallel:')
    pxmin = S1.getMins(map, 'parallel')
    pxmax = S1.getMaxs(map, 'parallel')
    print('pmin:', pxmin)
    print('pmax:', pxmax)

    if (mins != xmin).all():
        print('(min) getMin() is not equivalent to getRanges()')
    if (maxs != xmax).all():
        print('(max) getMax() is not equivalent to getRanges()')
    if (pxmin != xmin).all():
        print('(min) getMins() single is not equivalent to parallel')
    if (pxmax != xmax).all():
        print('(max) getMaxs() single is not equivalent to parallel')

if __name__ == '__main__':
    main()
