#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.rstar import RStar

def main():
    dim = 2
    lb = -np.ones((dim, 1))
    ub = np.ones((dim, 1))
    RS = RStar(lb = lb, ub = ub)
    
    W = np.matrix(np.random.rand(dim, dim))
    b = np.matrix(np.random.rand(dim, 1))
    RS1 = RS.affineMap(W, b)
    print(RS.__repr__())
    print(RS)
 
if __name__ == '__main__':
    main()
