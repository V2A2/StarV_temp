import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/box")
sys.path.insert(0, "engine/set/zono")
sys.path.insert(0, "engine/nn/funcs/tansig")
from box import *
from zono import *
from tansig import *

class TestTanSigReachZonoApporx(unittest.TestCase):
    """
        Tests approximation of hyperbolic tanget function by a zonotope
    """
    def test_reach_zono_approx(self):
        """
            First, a zonotope is created using a Box.
            The created zonotope is affine mapped. 
            Then, an approximation of TanSig function is applied.
            
            Output:
                Zono -> approximated Zono
        """
        np.set_printoptions(precision=25)
        lb = np.array([-1, 1])
        ub = np.array([1, 2])

        B = Box(lb, ub)
        Z = B.toZono()
        print("Constructed zonotope:\n")
        print(Z.__repr__())

        W = np.array([[0.5, 1], [-1, 1]])
        Z_affine = Z.affineMap(W)
        print("Affine mapped zonotope:\n")
        print(Z_affine.__repr__())

        Z_activation = TanSig.reach_zono_approx(Z_affine)
        print("Zonotope after an approximation of hyperbolic tanget function:\n")
        print(Z_activation.__repr__())
        [lb, ub] = Z_activation.getRanges()
        print('lb: \n', lb)
        print('ub: \n', ub)
        
if __name__ == '__main__':
    unittest.main()