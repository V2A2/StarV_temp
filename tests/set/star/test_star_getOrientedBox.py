import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono")
from zono import *

class TestStarGetOrientedBox(unittest.TestCase):
    
    def test_getOrientedBox(self):
        # still working: need to modify getOrientedBox
        c1 = np.array([0, 0])
        V1 = np.array([[1, -1], [1, 1], [0.5, 0], [-1, 0.5]]) 
        Z1 = Zono(c1, V1.transpose())
        I1 = Z1.toStar()
        
        I2 = I1.getOrientedBox()
        print('I2: \n', I2.__repr__())
        print('\nI1 getRanges: \n', I1.getRanges())
        print('\nI2 getRanges: \n', I2.getRanges())
        I3 = I1.getBox()
        print('\nI3 getRange: \n', I3.getRanges())
    
if __name__ == '__main__':
    unittest.main()