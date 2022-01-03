#!/usr/bin/env python3
"""
Created on Tue Oct  5 14:09:03 2021

@author: Apala
"""

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

def main():
    lb = np.matrix('-1; -1; -1')
    ub = np.matrix('1; 1; 1')

    B = Box(lb, ub)
    Z = B.toZono()
    print(Z.__repr__)

if __name__ == '__main__':
    main()
    