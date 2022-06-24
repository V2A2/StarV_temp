#!/usr/bin/python3
import numpy as np

import sys, copy

sys.path.insert(0, "engine/set/imagestar")
sys.path.insert(0, "engine/set/star")
sys.path.insert(0, "engine/set/zono")
sys.path.insert(0, "engine/set/box")
from imagestar import *
from star import *
from zono import *
from box import *

class ImageZono:
    """
        Class for representing set of images using Zonotope
        An image can be attacked by bounded noise. An attacked image can
        be represented using an ImageZono Set 
        author: Sung Woo Choi
        date: 9/30/2021

                            2-Dimensional ImageZono
        ====================================================================
                        Definition of 2-Dimensonal ImageZono
        
        A ImageZono Z= <c, V> is defined by: 
        S = {x| x = c + a[1]*V[1] + a[2]*V[2] + ... + a[n]*V[n]
                = V * b, V = {c V[1] V[2] ... V[n]}, 
                        b = [1 a[1] a[2] ... a[n]]^T                                   
                        where -1 <= a[i] <= 1}
        where, V[0], V[i] are 2D matrices with the same dimension, i.e., 
        V[i] \in R^{m x n}
        V[0] : is called the center matrix and V[i] is called the basic matrix 
        [a[1]...a[n] are called predicate variables
        The notion of 2D ImageZono is more general than the original Zonotope where
        the V[0] and V[i] are vectors. 
        
        Dimension of 2D ImageZono is the dimension of the center matrix V[0]
        
        ====================================================================
        The 2D representation of ImageZono is convenient for reachability analysis
    """
    
    def __init__(self, *args):
        """
            Constructor using 2D representation of an ImageZono
            V is an array of basis images in form of [height, width, numChannels, numPreds+1]
            height is height of image
            width is width of image
            numChannels is number of channels (e.g. color images have 3 channels)
            numPreds is number of predicate variables
            lb_image is lower bound of attack (high-dimensional numpy array)
            ub_image is upper bound of attack (high-dimensional numpy array)
        """
        from star import Star
        from zono import Zono

        # Initialize ImageZono properties with empty numpy sets and zero values.
        self.V = np.array([]) 
        [self.height, self.width, self.numChannels] = [np.array([]), np.array([]), np.array([])]
        self.numPreds = 0
        [self.lb_image, self.ub_image] = [np.array([]), np.array([])]
        
        length_args = len(args)
        if length_args == 2:
            [self.lb_image, self.ub_image] = copy.deepcopy(args)

            [lb_shape, ub_shape] =  [self.lb_image.shape, self.ub_image.shape]
            assert lb_shape == ub_shape, 'error: Inconsistent shape between lb_image and ub_image'
            assert len(lb_shape) == len(ub_shape) == (3 or 2), \
                'error: Inconsistent dimension between lb_image and ub_image; they should be 3D or 2D numpy array'
                
            self.height = lb_shape[0]
            self.width = lb_shape[1]
            if len(lb_shape) == 2:  self.numChannels = 1
            else:                   self.numChannels = lb_shape[2]
            
            # get basis images array
            lb = self.lb_image.flatten('F') # flatten in column-major (Fortan-style) order
            ub = self.ub_image.flatten('F')
            
            S = Star(lb, ub)
            self.numPreds = S.nVar
            self.V = S.V.reshape([self.height, self.width, self.numChannels, self.numPreds+1], order='F')
            
        elif length_args == 1:
            [self.V] = copy.deepcopy(args)
            [self.height, self.width, self.numChannels, self.numPreds] = self.V.shape
            self.numPreds -= 1
            
            N = self.height*self.width*self.numChannels
            center = self.V[:,:,:,0].flatten('F') # flatten in column-major (Fortan-style) order
            generators = self.V[:,:,:,1:self.numPreds+1].reshape(N, self.numPreds, order='F')
            
            Z = Zono(center, generators)
            [lb, ub] = Z.getBounds()
            
            self.lb_image = lb.reshape([self.height, self.width, self.numChannels], order='F')
            self.ub_image = ub.reshape([self.height, self.width, self.numChannels], order='F')
            
        elif length_args == 0:
            # create empty ImageZono (for preallocation an array of ImageZono)
            pass
            
        else:
            raise Exception('error: Invalid number of input arguments (should be 0, 1 or 2)')

    def evaluate(self, pred_val = np.array([])):
        """
            Evaluates an ImageZono with specific values of predicates
            pred_val: valued vector of predicate variables
        """
        assert isinstance(pred_val, np.ndarray), 'error: Center vector is not a numpy ndarray'
        assert len(pred_val.shape) == 1, 'error: Invalid predicate vector. It should be 1D numpy array'
        assert self.V.size, 'error: ImageZono is an empty set'
        assert pred_val.shape[0] == self.numPreds, 'error: Inconsistency between the size of the predicate vector and the number of predicates in the ImageZono'

        # check if all values of predicate variables are in [-1, 1]
        for i in range(self.numPreds):
            assert pred_val[i] <= 1 and pred_val[i] >= -1, 'error: Predicate values should be in the range of [-1, 1] for ImageZono'

        image = np.zeros([self.height, self.width, self.numChannels])
        for i in range(self.numChannels):
            image[:,:,i] = self.V[:,:,i,0]
            for j in range(1, self.numPreds + 1):
                image[:,:,i] += pred_val[j-1] * self.V[:,:,i,j]
        return image

    def affineMap(self, scale, offset):
        """
            Affine mapping of an ImageZono is another ImageZono
            y = scale * x + offset
            
            scale: scale coefficient [1 x 1 x NumChannels] numpy array
            offset: offset coefficient [1 x 1 x NumChannels] numpy array
            
            return -> a new ImageZono
        """
        scalar = np.isscalar(scale)
        if not scalar:
            assert scale.size and scale.shape[2] == self.numChannels and len(scale.shape) == 3, \
                'error: Inconsistent number of channels between scale array and the ImageZono'
        
        if self.numPreds > 0 and not scalar:
            scale = np.repeat(scale[:,:,:,np.newaxis], self.numPreds+1, axis=3) 
            
        if scalar:
            new_V = scale * self.V
        else:
            if scale.size:
                new_V = scale * self.V
            else:
                new_V = self.V

        if np.isscalar(offset):
            new_V[:,:,:,0] += offset
        else:
            if offset.size:
                new_V[:,:,:,0] += offset

        return ImageZono(new_V)

    def toZono(self):
        """
            Transforms ImageZono to Zono set
            
            return -> Zono
        """
        from zono import Zono

        center = self.V[:,:,:,0].flatten()
        generators = self.V[:,:,:,1:self.numPreds+1].reshape([self.height*self.width*self.numChannels, self.numPreds], order='F')
        return Zono(center, generators)

    def toImageStar(self):
        """
            Transforms ImageZono to ImageStar set
            
            return -> ImageStar
        """
        from imagestar import ImageStar
        pred_lb = -np.ones(self.numPreds)
        pred_ub = np.ones(self.numPreds)

        I = np.eye(self.numPreds)
        C = np.vstack([I,-I])            
        d = np.hstack([pred_ub, -pred_lb])
        return ImageStar(self.V, C, d, pred_lb, pred_ub, self.lb_image, self.ub_image)

    def contains(self, image):
        """
            Checks if an ImageZono contains an image
            image : input image
            
            output : 
                bool :  True -> if the ImageStar contain an image
                        False -> if the ImageStar does not contain an image
        """
        assert isinstance(image, np.ndarray), 'error: x is not numpy.ndarray'
         
        n = image.shape
        if len(n) == 2: # one channel image
            assert n[0] == self.height and n[1] == self.width and self.numChannels == 1, 'error: Inconsistent dimenion between input image and the ImageStar'
            y = image.flatten('F')
        elif len(n) == 3:
            assert  n[0] == self.height and n[1] == self.width and n[2] == self.numChannels, 'error: Inconsistent dimenion between input image and the ImageStar'
            y = image.flatten('F')
        else:
            raise Exception('error: invalid input image')

        Z = self.toZono()
        return Z.contains(y)
    
    def getRanges(self):
        """
            Gets lower and upper bounds (ranges) of ImageZono image
        """
        return [self.lb_image, self.ub_image]

    def is_p1_larger_p2(self, p1, p2):
        """
            Checks if a point in ImageZono is larger than the other point.
            Given two specific points in the ImageZono, the function checks if p1 > p2 is feasible.

            p1 : the first point = [h1, w1, c1]
            p2 : the second point = [h2, w2, c2]
            h: height, w: width, c: channel index
            
            return:
                True -> if p1 > p2 is feasible
                False -> p1 > p2 is not feasible
        """
        IS = self.toImageStar()
        return IS.is_p1_larger_p2(p1, p2)
        
    def __str__(self):
        print('class: %s' % (self.__class__))
        print('height: %s \nwidth: %s \nnumChannels: %s' % (self.height, self.width, self.numChannels))
        print('lb_image: [%s %s]' % (self.lb_image.shape, self.lb_image.dtype))
        print('ub_image: [%s %s]' % (self.ub_image.shape, self.ub_image.dtype))
        if len(self.V.shape) == 4:
            print('V: [%s %s]' % (self.V.shape, self.V.dtype))
        else:
            print('V: [%s %s]' % (self.V.shape, self.V.dtype))
        return 'numPreds: %s\n' % (self.numPreds)
    
    def __repr__(self, mat_ver=True):
        """
            mat_ver :   1 -> print images in [[[height, width] x channel] x (numPred+1)]
                        0 -> print images in [height x [width x [channel, (numPred+1)]]]
        """
        if mat_ver == False:
            return "class: %s \nheight: %s\nwidth: %s\nnumChannels: %s\nlb_image:\n%s\nub_image: \n%s\nV: \n%s\nnumPred: %s" % \
                (self.__class__, self.height, self.width, self.numChannels, self.lb_image, self.ub_image, self.V, self.numPreds)
        else:
            return "class: %s \nnumChannels: %s\nheight: %s\nwidth: %s\nlb_image:\n%s\nub_image: \n%s\nV: \n%s\nnumPred: %s" % \
                (self.__class__, self.height, self.width, self.numChannels, self.lb_image.transpose([-1, 0, 1]), self.ub_image.transpose([-1, 0, 1]), self.V.transpose([3,2,0,1]), self.numPreds)