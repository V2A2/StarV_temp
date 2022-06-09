import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box/")
from box import *

class TestBoxPartition(unittest.TestCase):
    """
        Tests Partition() function that partitions a Box into smaller boxes
    """
    
    def test_singlePartition(self):
        """
            Tests with initializing Box based on:
                lb : lower bound vector (1D numpy array)
                ub : upper bound vector (1D numpy array)
                
            Output:
                np.array([Boxes]) -> a numpy array of Boxes
        """
        lb = -np.ones(3)
        ub = np.ones(3)
        B = Box(lb, ub)
        print(B.__repr__())
        
        part_indexes = np.array([0, 1])
        part_numbers = np.array([10, 10])
        B1 = B.partition(part_indexes, part_numbers)
        n = len(B1)
        for i in range(n):
            print("%d: %s" % (i, B1[i].__repr__()))      
        
if __name__ == '__main__':
    unittest.main()