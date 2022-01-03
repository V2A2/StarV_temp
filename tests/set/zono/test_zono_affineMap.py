#!/usr/bin/python3

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.zono import Zono

def main():
    dim = 4
    C = np.ones(dim).reshape(-1,1)
    V = np.eye(dim)
    Z = Zono(C,V)
    
    W = np.random.rand(dim,dim)
    b = np.random.rand(dim,1)

    print('W: \n', W)
    print('b: \n', b)

    Z1 = Z.affineMap(W,b)
    print(Z1)
    print(Z1.__repr__)  


if __name__ == '__main__':
    main()   