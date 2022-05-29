#!/usr/bin/python3
import numpy as np

import sys
sys.path.insert(0, "../../../engine/set/")
from star import *

class Zono:
    # Zonotope class
    #   Z = (c , <v1, v2, ..., vn>) = c + a1 * v1 + ... + an * vn, 
    #   where -1 <= ai <= 1
    #   c is center, vi is generator
    # author: Sung Woo Choi
    # date: 9/26/2021

    def __init__(obj, c, V):
        # @c: center vector (np.matrix)
        # @V: generator matrix (np.matrix)

        assert isinstance(c, np.ndarray), 'error: center vector is not an ndarray'
        assert isinstance(V, np.ndarray), 'error: generator matrix is not an ndarray'
        #assert c.shape[1] == 1, 'error: center vector should be a column vector'
        assert c.shape[0] == V.shape[0], 'error: inconsistent dimension between a center vector and a generator matrix'

        obj.c = c
        obj.V = V
        obj.dim = V.shape[0]

    # affine mapping of a zonotope: Wz + b
    def affineMap(obj, W, b = np.array([])):
        # @W: mapping matrix
        # @b: mapping vector
        # return a new Zono

        assert isinstance(W, np.ndarray), 'error: weight matrix is not an matrix'
        assert isinstance(b, np.ndarray), 'error: bias vector is not an matrix'
        assert W.shape[1] == obj.dim, 'error: inconsistent dimension between weight matrix with the zonotope dimension'

        if b.size:
            assert b.shape[1] == 1, 'error: bias vector should be a column vector'
            assert W.shape[0] == b.shape[0], 'error: inconsistency between weight matrix and bias vector'           
            new_c = W @ obj.c + b
            new_V = W @ obj.V
        else:
            new_c = W @ obj.c
            new_V = W @ obj.V

        return Zono(new_c, new_V)

    # convert to Star
    def toStar(obj):
        #from engine.set.star import Star
        n = obj.V.shape[1]
        lb = -np.ones((n, 1))
        ub = np.ones((n, 1))

        C = np.vstack((np.eye(n), -np.eye(n)))
        d = np.vstack((np.ones((n,1)), np.ones((n,1))))
        V = np.hstack((obj.c, obj.V))
        return Star(V, C, d, lb, ub, outer_zono = obj)

    # convert to ImageZono
    def toImageZono(obj, height, width, numChannels):
        # @height: height of the image
        # @width: width of the image
        # @numChannels: number of channels of the image
        # return: ImageZono
        from engine.set.imagezono import ImageZono

        assert height*width*numChannels == obj.dim, 'error: inconsistent dimension, please change the height, width and numChannels to be consistent with the dimension of the zonotope' 

        new_V = np.hstack((obj.c, obj.V))
        numPreds = obj.V.shape[1]
        new_V = new_V.reshape((numPreds+1, numChannels, height, width))
        return ImageZono(new_V)

    # convert to ImageStar
    def toImageStar(obj, height, width, numChannels):
        # @height: height of the image
        # @width: width of the image
        # @numChannels: number of channels of the image
        # return: ImageStar
        im1 = obj.toStar()
        return im1.toImageStar(numChannels, height, width)


    # get a box bounding the zonotope
    def getBox(obj):
        from engine.set.box import Box
        
        lb = np.zeros((obj.dim, 1))
        ub = np.zeros((obj.dim, 1))
        for i in range(obj.dim):
            lb[i] = obj.c[i] - np.linalg.norm(obj.V[i, :], np.inf)
            ub[i] = obj.c[i] + np.linalg.norm(obj.V[i, :], np.inf)
        return Box(lb, ub)

    def getBounds(obj):
        # clip method from Stanley Bak to get bound of a zonotope
        pos_mat = obj.V.transpose()
        neg_mat = obj.V.transpose()
        pos_mat = np.where(neg_mat > 0, neg_mat, 0)
        neg_mat = np.where(neg_mat < 0, neg_mat, 0)
        pos1_mat = np.ones((1, obj.V.shape[1]))
        ub = np.transpose(pos1_mat @ (pos_mat - neg_mat))
        lb = -ub

        ub = obj.c + ub
        lb = obj.c + lb
        return [lb, ub]

    # get ranges of a zonotope
    def getRanges(obj):
        B = obj.getBox()
        return [B.lb, B.ub]

    # get range of a zonotope at specific index
    def getRange(obj, index):
        # @index: index of the state x[index]
        # return lb: lower bound of x[index] and 
        #        ub: upper bound of x[index]

        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid idnex')
        
        lb = obj.c[index] - np.norm(obj.V[index, :], np.inf)
        ub = obj.c[index] + np.norm(obj.V[index, :], np.inf)
        return [lb, ub]

#------------------check if this function is working--------------------------------------------
    # check if a index is larger than other
    def is_p1_larger_than_p2(obj, p1_id, p2_id):
        # @p1_id: index of point 1
        # @p2_id: index of point 2
        # return = 1 if there exists the case that p1 >= p2
        #          2 if there is no case that p1 >= p2
        S = obj.toStar()
        return S.is_p1_larger_than_p2(p1_id, p2_id)

    def __str__(obj):
        print('class: %s' % obj.__class__)
        print('c: [%sx%s %s]' % (obj.c.shape[0], obj.c.shape[1], obj.c.dtype))
        print('V: [%sx%s %s]' % (obj.V.shape[0], obj.V.shape[1], obj.V.dtype))
        return "dim: %s\n" % (obj.dim)

    def __repr__(obj):
        return "class: %s \nc: %s \nV: %s \ndim: %s \n" % (obj.__class__, obj.c, obj.V, obj.dim)