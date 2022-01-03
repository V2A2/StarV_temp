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



from engine.set.star import Star


def main():
    
   
    
    lb = np.matrix('1;1')
    ub = np.matrix('2;2')  
   

    S2 = Star(lb=lb,ub=ub)
    
    R = S2.is_p1_larger_than_p2(1,1)
    print(R)  
   
    


if __name__ == '__main__':
    main()   