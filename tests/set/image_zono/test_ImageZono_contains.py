import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/imagezono/")
from imagezono import *

class TestImageZonoEvaluate(unittest.TestCase):
    """
        Tests evaluate() function that evaluates an ImageZono with specific values of predicates
    """
    
    def test_evaluate(self):
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
        
        UB[:,:,0] = np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,2] = UB[:,:,1]
        
        IZ = ImageZono(LB, UB)
        print('\nPrint all information of IZ in detail: \n')
        print('\nIZ: ', IZ.__repr__())
        print('\n\nPrint inormation of ImageZono in short: \n')
        print('\nIZ: ', IZ.__str__())
        
        # image_2D = 1 - 2*np.random.rand(IZ.height, IZ.width)
        # print('\nDoes IZ contain randomely generated image_2D? ', IZ.contains(image_2D))
        
        image_3D = 1 - 2*np.random.rand(IZ.height, IZ.width, IZ.numChannels)
        print('\nDoes IZ contain randomely generated image_3D? ', IZ.contains(image_3D))

if __name__ == '__main__':
    unittest.main()