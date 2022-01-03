#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box
from engine.set.rstar import RStar

def main():
    dim = 3
    lb = -np.ones((dim, 1))
    ub = np.ones((dim, 1))

    RS = RStar(lb = lb, ub = ub)
    print('Print all information of rstar in detail:')
    print(RS.__repr__())
    print('\nPrint inormation of rstar in short:')
    print(RS)

if __name__ == '__main__':
    main()