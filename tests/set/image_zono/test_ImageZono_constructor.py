import unittest

import sys
import numpy as np

sys.path.insert(0, "engine/set/imagezono/")
from imagezono import *

class TestImageZonoConstructor(unittest.TestCase):
    """
        Tests ImageZono constructor
    """
    
    def test_bounds_init(self):
        """
            Tests the initialization of ImageZono with attack bounds:
                LB : lower bound of attack (high-dimensional numpy array)
                UB : upper bound of attack (high-dimensional numpy array)
                
            Output:
                ImageZono :
                    V -> an array of basis images
                    height -> height of image
                    width -> width of image
                    numChannels -> number of channels (e.g. color images have 3 channels)
                    numPreds -> number of predicate variables
                    lb_image -> lower bound of attack (high-dimensional numpy array)
                    ub_image -> upper bound of attack (high-dimensional numpy array)
        """
        LB = np.zeros([4, 4, 3])
        UB = np.zeros([4, 4, 3])
    
        # attack on piexel (1,1) and (1,2)
        LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        LB[:,:,2] = LB[:,:,1]
        print("\nLB:\n", LB.transpose([-1, 0, 1]))
        print("\nLB.shape:\n", LB.shape)
        
        UB[:,:,0] = np.array([[0, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        UB[:,:,2] = UB[:,:,1]
        print("\nUB:\n", UB.transpose([-1, 0, 1]))
        print("\nUB:\n", UB.shape)
        
        image = ImageZono(LB, UB)
        print('\nPrint all information of ImageZono in detail: \n')
        print('\nimage: ', image.__repr__())
        print('\n\nPrint inormation of ImageZono in short: \n')
        print('\nimage: ', image.__str__())
        
    def test_basis_images_init(self):
        """
            Tests the initialization of ImageZono with basis images:
                V -> an array of basis images in form of [height, width, numChannels, numPreds+1]
                
            Output:
                ImageZono :
                    V -> an array of basis images
                    height -> height of image
                    width -> width of image
                    numChannels -> number of channels (e.g. color images have 3 channels)
                    numPreds -> number of predicate variables
                    lb_image -> lower bound of attack (high-dimensional numpy array)
                    ub_image -> upper bound of attack (high-dimensional numpy array)
        """
        height = 4
        width = 4
        numChannels = 3
        numPreds = 7
        V = np.zeros([height, width, numChannels, numPreds])
        V[:,:,0,0] = np.array([[-0.05, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,0,1] = np.array([[0.05, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,0,2] = np.array([[0, 0.2, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,1,3] = np.array([[0.1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,1,4] = np.array([[0, 0.15, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,2,5] = np.array([[0.1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,2,6] = np.array([[0, 0.15, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
        V[:,:,2,6] = np.array([[0.1, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 0]])
                
        image = ImageZono(V)
        print('\nPrint all information of ImageZono in detail: \n')
        print('\nimage: ', image.__repr__())
        print('\n\nPrint inormation of ImageZono in short: \n')
        print('\nimage: ', image.__str__())

if __name__ == '__main__':
    unittest.main()