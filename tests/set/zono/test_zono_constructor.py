import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/zono/")
from zono import *

class TestZonoConstructor(unittest.TestCase):
    """
        Tests Zono constructor
    """
    
    def test_basic_init(self):
        """
            Tests the initialization with:
                c : center vector (1D numpy array)
                V : generator matrix (2D numpy array)
                
            Output:
                Zono ->
                    c -> center vector (1D numpy array)
                    V -> generator matrix (2D numpy array)
        """
        c = np.zeros(2)
        V = np.array([[1, -1], [1, 1]])

        Z = Zono(c, V)
        print(Z.__repr__())
        print(Z.__str__())
        Z.plot()

if __name__ == '__main__':
    unittest.main()