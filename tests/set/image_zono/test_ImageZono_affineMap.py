import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/imagezono/")
from imagezono import *

class TestImageZonoAffineMap(unittest.TestCase):
    """
        Tests affine mapping of ImageZono
    """
    
    def test_affineMap(self):
        """
            Initiate the ImageZono with lower and upper bounds of attack (high-dimensional numpy arrays)
            Affine mapping of an ImageZono is another ImageZono
            y = scale * x + offset
        
            scale: scale coefficient [1 x 1 x NumChannels] numpy array
            offset: offset coefficient [1 x 1 x NumChannels] numpy array
            
            Output -> ImageZono
        """
        LB = np.zeros([4, 4, 3])
        UB = np.zeros([4, 4, 3])
    
        # attack on piexel (1,1) and (1,2)
        LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,2] = LB[:,:,1]
        
        UB[:,:,0] = np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,2] = UB[:,:,1]
        
        image = ImageZono(LB, UB)
        print('\nPrint all information of ImageZono in detail: \n')
        print('\nimage: ', image.__repr__())
        print('\n\nPrint inormation of ImageZono in short: \n')
        print('\nimage: ', image.__str__())

        scale = 2*np.ones([1,1,3]) 
        offset = np.zeros([1,1,3])

        new_image = image.affineMap(scale, offset)
        print('\nPrint all information of new_image in detail: \n')
        print('\nimage: ', new_image.__repr__())
        print('\n\nPrint inormation of new_image in short: \n')
        print('\nimage: ', new_image.__str__())
        
if __name__ == '__main__':
    unittest.main()