#!/usr/bin/python3
import numpy as np

class ImageZono:
    # ImageZono class
    # Class for representing set of images using Zonotope
    # An image can be attacked by bounded noise. An attacked image can
    # be represented using an ImageZono Set 
    # author: Sung Woo Choi
    # date: 9/30/2021

    #                      2-Dimensional ImageZono
    # ====================================================================%
    #                   Definition of 2-Dimensonal ImageZono
    # 
    # A ImageZono Z= <c, V> is defined by: 
    # S = {x| x = c + a[1]*V[1] + a[2]*V[2] + ... + a[n]*V[n]
    #           = V * b, V = {c V[1] V[2] ... V[n]}, 
    #                    b = [1 a[1] a[2] ... a[n]]^T                                   
    #                    where -1 <= a[i] <= 1}
    # where, V[0], V[i] are 2D matrices with the same dimension, i.e., 
    # V[i] \in R^{m x n}
    # V[0] : is called the center matrix and V[i] is called the basic matrix 
    # [a[1]...a[n] are called predicate variables
    # The notion of 2D ImageZono is more general than the original Zonotope where
    # the V[0] and V[i] are vectors. 
    # 
    # Dimension of 2D ImageZono is the dimension of the center matrix V[0]
    # 
    # ====================================================================%
    # The 2D representation of ImageZono is convenient for reachability analysis

    def __init__(obj, V = np.array([]),    # an array of basis images    
                lb_image = np.array([]),   # lower bound of attack (high-dimensional array)
                ub_image = np.array([])):  # upper bound of attack (high-dimensional array)
        from engine.set.star import Star
        from engine.set.zono import Zono

        if V.size:
            assert isinstance(V, np.ndarray), 'error: an array of basis images is not an ndarray'

            obj.V = V
            [obj.numChannels, obj.height, obj.width] = obj.V[0].shape
            obj.numPreds = obj.V.shape[0] - 1

            center = obj.V[1,:,:,:]
            generators = obj.V[1:obj.numPreds + 1, :,:,:]
            center = center.reshape(-1,1)
            generators = generators.reshape(-1, obj.numPreds)

            Z = Zono(center, generators)
            [lb, ub] = Z.getBounds()

            # A box representation of an ImageZono
            # A convenient way for user to specify the attack
            obj.lb_image = np.array(lb).reshape((obj.numChannels, obj.height, obj.width))
            obj.ub_image = np.array(ub).reshape((obj.numChannels, obj.height, obj.width))
            return


        if lb_image.size and ub_image.size:
            assert isinstance(lb_image, np.ndarray), 'error: a lower bound of attack is not an ndarray'
            assert isinstance(ub_image, np.ndarray), 'error: a upper bound of attack is not an ndarray'

            if lb_image.shape != ub_image.shape:
                raise Exception('error: different sizes between lower bound image and upper bound image')

            obj.lb_image = lb_image
            obj.ub_image = ub_image

            if len(lb_image.shape) == 3:
                obj.numChannels = obj.lb_image.shape[0] # number of channels, e.g., color images have 3 channel
                obj.height = obj.lb_image.shape[1]      # height of image
                obj.width = obj.lb_image.shape[2]       # width of image
            elif len(lb_image.shape) == 2:
                obj.numChannels = 1
                obj.height = obj.lb_image.shape[0]
                obj.width = obj.lb_image.shape[1]
            else:
                raise Exception('image bounds need to be a tuple of three elements: numChannels, image width, image height')
            lb = obj.lb_image.reshape(-1,1)
            ub = obj.ub_image.reshape(-1,1)

            S = Star(lb=lb, ub=ub)
            obj.numPreds = S.nVar       # number of predicate variables
            obj.V = np.reshape(S.V, (obj.numPreds + 1, obj.numChannels, obj.height, obj.width))
            return
        
        raise Exception('error: failed to create ImageZono')

#------------------check if this function is working--------------------------------------------
    # evaluate an ImageZono with specific values of predicates
    def evaluate(obj, pred_val = np.matrix([])):
        # @pred_val: valued vector of predicate variables

        assert obj.V.size, 'error: the ImageZono is an empty set'
        assert pred_val.size[1] == 1, 'error: invalid predicate vector'
        assert pred_val.size[0] == obj.numPreds, 'error: inconsistency between the size of the predicate vector and the number of preeicates in the ImageZono'

        # check if all values of predicate variables are in [-1, 1]
        for i in range(obj.numPreds):
            if not (pred_val[i]<=1 and pred_val[i]>=-1):
                raise Exception('error: predicate values should be in the range of [-1, 1] for ImageZono')

        image = np.zeros((obj.numChannels, obj.height, obj.width))
        for i in range(obj.numChannels):
            image[i, :, :] = obj.V[1, i, :, :]
            for j in range(1, obj.numPreds + 1):
                image[i, :, :] = image[i, :, :] + pred_val[j-1] * obj.V[j, i, :, :]
        return image

    # affineMap of an ImageZono is another ImageZono
    # y = scale * x + offset
    def affineMap(obj, scale, offset):
        # @scale: scale coefficient [1 x 1 x NumChannels] array
        # @offset: offset coefficient [1 x 1 x NumChannels] array
        # return: a new ImageZono

        assert scale.size and not np.isscalar(scale) and scale.shape[0] == obj.numChannels, 'error: inconsistent number of channels between scale array and the ImageZono'
        
        if scale.size:
            new_V = scale * obj.V
        else:
            new_V = obj.V

        if offset.size:
            new_V[1, :, :, :] = new_V[1, :, :, :] + offset
        
        return ImageZono(new_V)

    # transform to Zono
    def toZono(obj):
        from engine.set.zono import Zono

        center = obj.V[1,:,:,:,]
        generators = obj.V[1:obj.numPreds + 1,:,:,:]

        center = center.reshape(-1, 1)
        generators = np.reshape(generators, (obj.height*obj.width*obj.numChannels, obj.numPreds))
        return Zono(center, generators)

#------------------check if this function is working--------------------------------------------
    # transform to ImageStar
    def toImageStar(obj):
        from imagestar import ImageStar
        pred_lb = -np.ones((obj.numPreds, 1))
        pred_ub = np.ones((obj.numPreds, 1))

        C = np.hstack((np.eye(obj.numPreds), -np.eye(obj.numPreds)))            
        d = np.hstack((pred_ub, -pred_lb))
        return ImageStar(obj.V, C, d, pred_lb, pred_ub, obj.lb_image, obj.ub_image)
    

#------------------check if this function is working--------------------------------------------
    # contain, check if an ImageZono contain an image
    def contains(obj, image):
        # @image: input image
        # @bool = 1 if the ImageStar contain the image
        #         2 if the ImageStar does not contain the image

        n = image.shape
        if len(n) == 2: # one channel image
            assert obj.numChannels == 1 and n[1] == obj.height and n[2] == obj.width, 'error: inconsistent dimenion between input image and the ImageStar'
            y = image.flatten()
        elif len(n) == 3:
            assert n[0] == obj.numChannels and n[1] == obj.height and n[2] == obj.width, 'error: inconsistent dimenion between input image and the ImageStar'
            y = image.flatten()
        else:
            raise Exception('error: invalid input image')

        Z = obj.toZono()
        return Z.contains(y)

    # get Ranges
    def getRanges(obj):
        return [obj.lb_image, obj.ub_image]

#------------------check if this function is working--------------------------------------------
    def is_p1_larger_p2(obj, p1, p2):
        # @p1: the first point = []
        # @p2: the second point = []
        # h: height, w: width, c: channel index

        # @b = 1 -> p1 > p2 is feasible
        #    = 0 -> p1 > p2 is infeasible

        S = obj.toImageStar
        return  S.is_p1_larger_p2(p1, p2)
        
    def __str__(obj):
        print('class: %s' % (obj.__class__))
        print('height: %s \nwidth: %s' % (obj.height, obj.width))
        print('lb_image: [%sx%sx%s %s]' % (obj.lb_image.shape[0], obj.lb_image.shape[1], obj.lb_image.shape[2], obj.lb_image.dtype))
        print('ub_image: [%sx%sx%s %s]' % (obj.ub_image.shape[0], obj.ub_image.shape[1], obj.ub_image.shape[2], obj.ub_image.dtype))
        if len(obj.V.shape) == 4:
            print('V: [%sx%sx%sx%s %s]' % (obj.V.shape[0], obj.V.shape[1], obj.V.shape[2], obj.V.shape[3], obj.V.dtype))
        else:
            print('V: [%sx%sx%s %s]' % (obj.V.shape[0], obj.V.shape[1], obj.V.shape[2], obj.V.dtype))
        return 'numPreds: %s\n' % (obj.numPreds)
    
    def __repr__(obj):
        return "class: %s \nnumChannels: %s\nheight: %s\nwidth: %s\nlb_image:\n%s\nub_image: \n%s\nV: \n%s\nnumPred: %s" % (obj.__class__, obj.numChannels, obj.height, obj.width, obj.lb_image, obj.ub_image, obj.V, obj.numPreds)




