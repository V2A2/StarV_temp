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
from engine.set.box import Box

def main():
    lb = np.matrix('49; 25; 9; 20')
    ub = np.matrix('51; 25.2; 11; 20.2')
    
    S = Star(lb = lb, ub = ub)
    print('S: \n', S.__repr__())
    
    B = S.getBox()
    print('\nB: \n', B.__repr__())
    
    
if __name__ == '__main__':
    main()