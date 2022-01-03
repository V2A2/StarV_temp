#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.box import Box

def main():
    lb = np.matrix('-3; -3')
    ub = np.matrix('2; 2')  

    S = Star(lb = lb, ub = ub)
    print('S: \n', S.__repr__())
    
    Z = S.getZono()
    print('\nZ: \n', Z.__repr__())
    
if __name__ == '__main__':
    main()    