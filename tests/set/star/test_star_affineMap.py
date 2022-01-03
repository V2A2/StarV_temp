#!/usr/bin/python3
"""
Created on Tue Oct  5 16:35:49 2021

@author: Apala
"""
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star

def main():
    lb = np.matrix('1; 1')
    ub = np.matrix('2; 2')
    
    S = Star(lb = lb, ub = ub)
    print('S: ', S.__repr__())
     
    V = S.V
    C = S.C
    d = S.d

    S2 = Star(V, C, d)
    print('\nS2: ', S2.__repr__())
    
    W = np.matrix('1 -1; 1 1')
    b = np.matrix('0.5; 0.5')
    
    Sa = S2.affineMap(W, b)
    print('\nSa: ', Sa.__repr__())    

    
if __name__ == '__main__':
    main()
