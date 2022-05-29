#!/usr/bin/python3
import numpy as np

import sys
sys.path.insert(0, "../../../engine/set/")

from zono import *

class Box:
    # Hyper-rectangle class
    # Box and simple methods
    # author: Sung Woo Choi
    # date: 9/21/2021

    # constructor
    def __init__(obj, lb = np.array([]), ub = np.array([])):
        # @lb: lower-bound vector
        # @ub: upper-bound vector

        assert isinstance(lb, np.ndarray), 'error: lower bound vector is not an ndarray'
        assert isinstance(ub, np.ndarray), 'error: upper bound vector is not an ndarray'
        #assert lb.shape[1] == ub.shape[1] == 1, 'error: lb and ub should be a column vector'
        assert lb.shape[0] == ub.shape[0], 'error: inconsistent dimensions between lb and ub'

        obj.lb = lb
        obj.ub = ub
        obj.dim = lb.shape[0]

        obj.center = 0.5 * (ub + lb)
        obj.generators = np.array([])
        vec = 0.5 * (ub - lb)
        for i in range(obj.dim):
            if vec[i] != 0:
                gen = np.zeros((obj.dim, 1))
                gen[i] = vec[i]
                obj.generators = np.hstack([obj.generators, gen]) if obj.generators.size else gen
        
        if np.linalg.norm(vec, 2) == 0:
            obj.generators = np.zeros((obj.dim, 1))

    # affine mapping of a box
    def affineMap(obj, W, b = np.array([])):
        # @W: mapping matrix
        # @b: mapping vector
        # return a new Box

        assert isinstance(W, np.ndarray), 'error: weight matrix is not an matrix'
        assert isinstance(b, np.ndarray), 'error: bias vector is not an matrix'
        assert W.shape[1] == obj.generators.shape[0], 'error: inconsistent dimension between weight matrix with the box dimension'

        if b.size:
            assert b.shape[1] == 1, 'error: bias vector should be a column vector'
            assert W.shape[0] == b.shape[0], 'error: inconsistency between weight matrix and bias vector'
            new_center = W @ obj.center + b
            new_generators = W @ obj.generators
        else:
            new_center = W @ obj.center
            new_generators = W @ obj.generators

        n = new_center.shape[0]
        new_lb = np.zeros((n, 1))
        new_ub = np.zeros((n, 1))

        for i in range(n):
            v = new_generators[i, :].reshape(-1, 1)
            new_lb[i] = new_center[i] - np.linalg.norm(v, 1)
            new_ub[i] = new_center[i] + np.linalg.norm(v, 1)

        return Box(new_lb, new_ub)

    # transform box to star set
    def toStar(obj):
        Z = obj.toZono()
        return Z.toStar()

    # transform box to zonotope
    def toZono(obj):
        #from engine.set.zono import Zono
        return Zono(obj.center, obj.generators)

    # get Range
    def getRange(obj):
        return [obj.lb, obj.ub]
        
    def __str__(obj):
        print('class: %s' % obj.__class__)
        print('lb: [%sx%s %s]' % (obj.lb.shape[0], obj.lb.shape[1], obj.lb.dtype))
        print('ub: [%sx%s %s]' % (obj.ub.shape[0], obj.ub.shape[1], obj.ub.dtype))
        print('dim: %s' % obj.dim)
        print('center: [%sx%s %s]' % (obj.center.shape[0], obj.center.shape[1], obj.center.dtype))
        return 'generators: [%sx%s %s]\n' % (obj.generators.shape[0], obj.generators.shape[1], obj.generators.dtype)

    def __repr__(obj):
        return "class: %s \nlb: \n%s \nub: \n%s \ndim:%s \ncenter: \n%s \ngenerator: \n%s\n" % (obj.__class__, obj.lb, obj.ub, obj.dim, obj.center, obj.generators)
