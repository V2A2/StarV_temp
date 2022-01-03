#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

def main():
    lb = np.matrix('-1; -1; -1')
    ub = np.matrix('1; 1; 1')

    print('\n-----------------------input box------------------------')
    B = Box(lb,ub)
    print(B.__repr__())

    S = B.toStar()
    print('\n---------------------box toStar()-----------------------')
    print(S.__repr__())

    W = np.random.rand(3,3)
    b = np.random.rand(3,1)
    
    print('\n-------------------affine mapped box--------------------')
    B = B.affineMap(W,b)
    print(B.__repr__())
    print('\n---------------affine mapped box toStar()---------------')
    SB = B.toStar()
    print(SB.__repr__())
    print('\n-------------------affine mapped star-------------------')
    S = S.affineMap(W,b)
    print(S.__repr__())

    
if __name__ == '__main__':
    main()