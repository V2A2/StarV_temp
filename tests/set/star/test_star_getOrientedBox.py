#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.zono import Zono

def main():
    # still working: need to modify getOrientedBox
    c1 = np.matrix('0; 0')
    V1 = np.matrix('1 -1; 1 1; 0.5 0; -1 0.5')
    Z1 = Zono(c1, V1.transpose())
    I1 = Z1.toStar()
    
    I2 = I1.getOrientedBox()
    print('I2: \n', I2.__repr__())
    print('\nI2 getRanges: \n', I2.getRanges())
    I3 = I1.getBox()
    print('\nI3 getRange: \n', I3.getRange())
    
if __name__ == '__main__':
    main()