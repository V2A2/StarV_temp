#!/usr/bin/python3
"""
Created on Tue Oct  5 16:32:18 2021

@author: Apala
"""

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")


from engine.set.star import Star
from engine.set.box import Box

def main():
    
    lb = np.matrix('1;1')
    ub = np.matrix('2;2')
    
   

    S = Star(lb=lb, ub=ub)
    print(S)
    
    E = S.isEmptySet()
    print("Is star set empty?",E)
    
    """
     V = S.V
    C = S.C
    d = S.d

    S2 = Star(V, C, d)
    print(S2)"""
    
    
    
if __name__ == '__main__':
    main()