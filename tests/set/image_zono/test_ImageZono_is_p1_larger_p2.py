import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/imagezono/")
from imagezono import *

class TestImageZonoToZono(unittest.TestCase):
    """
        Tests affine mapping of ImageZono
    """
    
    def test_toZoono(self):
        """
            Initiate the ImageZono with lower and upper bounds of attack (high-dimensional numpy arrays)
            Convert ImageZono to Zono by toZono() function
            
            output -> Zono
        """
        LB = np.zeros([4, 4, 3])
        UB = np.zeros([4, 4, 3])

        # attack on piexel (1,1) and (1,2)
        LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,2] = LB[:,:,1]
        
        UB[:,:,0] = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,2] = UB[:,:,1]
        
        image = ImageZono(LB, UB)
        print('\nPrint all information of ImageZono in detail: \n')
        print('\nimage: ', image.__repr__())
        print('\n\nPrint inormation of ImageZono in short: \n')
        print('\nimage: ', image.__str__())
        
        p1 = np.array([0,0,0])
        p2 = np.array([2,3,1])
        print('\nIs p1 larger than p2 in image? ', image.is_p1_larger_p2(p1, p2))

if __name__ == '__main__':
    unittest.main()