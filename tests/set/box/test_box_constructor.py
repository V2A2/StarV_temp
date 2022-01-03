#!/usr/bin/env python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.box import Box

def main():
    lb = np.matrix('-1; -1')
    ub = np.matrix('1; 1')

    print('lb: ', lb)
    print('ub: ', ub)

    B = Box(lb,ub)
    print(B.__repr__())

if __name__ == '__main__':
    main()