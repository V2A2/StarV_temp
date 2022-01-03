#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.box import Box

def main():
    dim = 2
    lb = -2*np.ones((dim, 1))
    ub = 2*np.ones((dim, 1))
    
    S = Star(lb=lb, ub=ub)
    print('S: \n', S.__repr__())
    
    V = S.V
    C = S.C
    d = S.d
    pred_lb = S.predicate_lb
    pred_ub = S.predicate_ub
    
    S2 = Star(V, C, d, pred_lb, pred_ub)
    print('\nPrint all information of star in detail:')
    print('S2: \n', S2.__repr__())
    print('\n\nPrint inormation of star in short:')
    print('S2: \n', S2)

    
if __name__ == '__main__':
    main()