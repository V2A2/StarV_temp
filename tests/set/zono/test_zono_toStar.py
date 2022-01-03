#!/usr/bin/python3
"""
Created on Fri Oct  8 17:31:53 2021

@author: Apala
"""

import sys
import os
import numpy as np

os.chdir('tests/')
sys.path.append("..")



from engine.set.zono import Zono


def main():
    
   
    
    C = np.matrix('0;0')
    V = np.matrix('1 -1; 1 1')  
    S = Zono(C,V)
    
  
    Z = S.toStar()
 
    print(Z)  
   
    


if __name__ == '__main__':
    main()   