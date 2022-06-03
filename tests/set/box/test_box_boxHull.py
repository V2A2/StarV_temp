import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxBoxHull(unittest.TestCase):
    """
        Tests boxHull() function that  merges all boxes into a single box
    """
    
    def test_getVertices(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                Box ->
                    dim -> dimension of a Box
                    lb -> lower bound vector (1D numpy array)
                    ub -> upper bound vector (1D numpy array)
        """
        lb = np.array([-5, -5])
        ub = np.array([-4, -4])
        B1 = Box(lb, ub)
        
        lb = np.array([3, 4])
        ub = np.array([4, 5])
        B2 = Box(lb, ub)
        
        lb = np.array([-2, -3])
        ub = np.array([0, 5])
        B3 = Box(lb, ub)
        
        boxes = np.array([B1, B2, B3])
        result_box = Box.boxHull(boxes)
        print(result_box.__repr__())
        
if __name__ == '__main__':
    unittest.main()