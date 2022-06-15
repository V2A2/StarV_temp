import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star")
sys.path.insert(0, "engine/nn/funcs/tansig")
from star import *
from tansig import *

class TestTanSigReachStarApporx(unittest.TestCase):
    """
        Tests approximation of hyperbolic tanget function 
        with upto four linear constraints by a Star
    """
    def test_reach_absdom_approx(self):
        """
            First, a Star is created using lower and upper bounds.
            The created Star is affine mapped. 
            Then, an approximation of TanSig function is applied.
           
            Output:
                Star -> approximated Star
        """
        np.set_printoptions(precision=25)
        lb = np.array([-0.1,  -0.1])
        ub = np.array([0.1, 0.1])

        S = Star(lb, ub)
        print("Constructed star:\n")
        print(S.__repr__())
        
        [lb, ub] = S.getRanges()
        print("\nBounds of star:")
        print('lb: \n', lb)
        print('ub: \n', ub)
        
        # W = np.array([[0.1, -1], [0.1, 1]])
        # W = 1 - 2*np.random.rand(2, 2)
        # b = 1 - 2*np.random.rand(2)
        W = np.array([[-0.40051, 0.38174], [0.43668, 0.74866]])
        b = np.array([-0.66479, 0.38367])
        # W = np.array([[-0.02916, -0.37064], [-0.67113, -0.80929]])
        # b = np.array([-0.02452, -0.65121])
        print("W: ", W)
        print("b: ", b)
        
        S_affine = S.affineMap(W, b)
        print("Affine mapped star:\n")
        print(S_affine.__repr__())
        [lb, ub] = S_affine.getRanges()
        print("\nBounds of star:")
        print('lb: \n', lb)
        print('ub: \n', ub)

        S_activation = TanSig.reach_star_approx(S_affine)
        print("Star after an approximation of hyperbolic tanget function:\n")
        print(S_activation.__repr__())
        print("Is S_activation an emptySet? ", S_activation.isEmptySet())
        [lb, ub] = S_activation.getRanges()
        print("\nBounds of star:")
        print('lb: \n', lb)
        print('ub: \n', ub)

        # print('after relaxed reach_star_approx')
        # relaxFactor = 0.5
        # S_relax = TanSig.reach_star_approx(I, 'approx-star', relaxFactor, "", 'gurobi')
        # [lb, ub] = S_relax.getRanges()
        # print('relaxed lb: \n', lb)
        # print('relaxed ub: \n', ub)
        
if __name__ == '__main__':
    unittest.main()