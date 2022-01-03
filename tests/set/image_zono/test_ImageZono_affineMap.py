#!/usr/bin/python3
import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")

from engine.set.imagezono import ImageZono

def main():
    # attack on piexel (1,1) and (1,2)
    L1 = np.matrix('-0.1 -0.2 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0') 
    L2 = np.matrix('-0.1 -0.15 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
    L3 = L2
    LB = np.array([L1, L2, L3])

    U1 = np.matrix('0 0.2 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
    U2 = np.matrix('0.1 0.15 0 0; 0 0 0 0; 0 0 0 0; 0 0 0 0')
    U3 = U2
    UB = np.array([U1, U2, U3])

    image = ImageZono(lb_image = LB, ub_image = UB)
    print("image: ", image)

    scale = 2*np.ones((3,1,1))
    offset = np.zeros((3,1,1))

    new_image = image.affineMap(scale, offset)
    print("new_image: ", new_image)
if __name__ == '__main__':
    main()