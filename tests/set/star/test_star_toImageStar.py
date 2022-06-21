import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/star/")
from star import *

class TestStarToImageStar(unittest.TestCase):
    """
        Tests toImageStar() function that converts current Star to ImageStar set
    """

    def test_toImageStar(self):
        """
            Generate a Star set and convert it to ImageStar set
                
            Output -> ImageStar   
        """
        V1 = np.array([[0, 1, 0], [0, 0, 1]])
        C1 = np.array([[0.99985, 0.01706], [-0.40967, 0.91224], [-0.57369, -0.81907]])
        d1 = np.array([0.95441, 0.46047, -0.82643])
        predicate_lb_1 = np.array([0.43863, 0.34452])
        predicate_ub_1 = np.array([0.94867, 0.92634])
        S1 = Star(V1, C1, d1, predicate_lb_1, predicate_ub_1)
        print('\nPrint S1 in detail: \n')
        print(S1.__repr__())
        
        IS = S1.toImageStar(1, 1, S1.dim)
        print('\nPrint converted ImageStar in detail: \n')
        print('numChannel: ', IS.get_num_channel())
        print('height: ', IS.get_height())
        print('width: ', IS.get_width())
        print('V: ', IS.get_V())
        print('C: ', IS.get_C())
        print('d: ', IS.get_d())
        print('pred_lb: ', IS.get_pred_lb())
        print('pred_ub: ', IS.get_pred_ub())
    
if __name__ == '__main__':
    unittest.main()