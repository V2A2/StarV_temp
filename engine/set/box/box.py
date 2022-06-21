#!/usr/bin/python3
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

import sys, copy

sys.path.insert(0, "engine/set/zono")
from zono import *

class Box:
    """
        Class for representing a hyper-rectangle convex set using Box set
        Box and simple methods
        author: Sung Woo Choi
        date: 9/21/2021
        
        Representation of a Box
        ====================================================================
        A Box set B is defined by:
        B = {x| lb <= x <= ub},
        where lb is lower bound of a state vector x and
                ub is upper bound of a state vector x.
        
        center vector and generator matrix are supporting vector and matrix 
        of a Box. They are used to compute lower bound vector and upper of 
        bound vector of a state vector x.
        
        center: is a center vector
        generator: is a center matrix
        ====================================================================
    """

    def __init__(self, lb = np.array([]), ub = np.array([])):
        """
            Constructor a Box using lower bound vector and upper bound vector

            lb : lower-bound vector, 1D numpy array
            ub : upper-bound vector, 1D numpy array
        """
        assert isinstance(lb, np.ndarray), 'error: Lower bound vector is not a numpy ndarray'
        assert isinstance(ub, np.ndarray), 'error: Upper bound vector is not a numpy ndarray'
        assert lb.shape == ub.shape, 'error: Inconsistent dimension between lower bound vector and upper bound vector'
        assert len(lb.shape) == len(ub.shape) == 1, 'error: Lower bound vector and upper bound vector should be 1D numpy array'

        self.lb = lb
        self.ub = ub
        self.dim = lb.shape[0]

        self.center = 0.5 * (ub + lb)
        self.generators = np.array([])
        vec = 0.5 * (ub - lb)
        
        for i in range(self.dim):
            if vec[i] != 0:
                gen = np.zeros((self.dim, 1))
                gen[i] = vec[i]
                self.generators = np.hstack([self.generators, gen]) if self.generators.size else gen
        
        if np.linalg.norm(vec, 2) == 0:
            self.generators = np.zeros((self.dim, 1))

    def singlePartition(self, part_id, part_num):
        """
            A single partition of a Box
            part_id : index of the state being partitioned
            part_num : number of partition
            
            return -> a numpy array of Boxes
        """
        assert part_id >= 0 and part_id < self.dim, 'error: Invalid partition index'
        assert part_num > 0, 'error: Invalid partition number'
        
        if part_num == 1:
            return self
        else:
            del_ = (self.ub[part_id] - self.lb[part_id]) / part_num
            
            Bs = np.array([])
            for i in range(part_num):
                new_lb = copy.deepcopy(self.lb)
                new_ub = copy.deepcopy(self.ub)
                new_lb[part_id] = self.lb[part_id] + (i)*del_
                new_ub[part_id] = new_lb[part_id] + del_
                Bs = np.hstack([Bs, Box(new_lb, new_ub)])
            return Bs
        
    def partition(self, part_indexes, part_numbers):
        """
            Partition a Box into smaller boxes
            part_indexes : the indexes of the state begin partitioned (1D numpy array)
            part_number : the number of partitions at specific index (1D numpy array)
            
            return -> a numpy array of Boxes
        """
        assert len(part_indexes.shape) == len(part_numbers.shape) == 1, 'error: Invalid part_indexes array or part_numbers array. They should be 1D numpy array'
        assert part_indexes.shape == part_numbers.shape, 'error: Inconsistency between part_indexes and part_numbers arrays. They should have the same number of elements'

        n = len(part_indexes)
        for i in range(n):
            assert part_indexes[i] >= 0 and part_indexes[i] < self.dim, 'error: The %d-th index is part_indexes array is invalid' % i
            assert part_numbers[i] > 0, 'error: The %d-th number in part_numbers array is invalid' % i
            
        B1 = np.array([self])
        for i in range(n):
            m = len(B1)
            B2 = np.array([])
            for j in range(m):
                B2 = np.hstack([B2, B1[j].singlePartition(part_indexes[i], part_numbers[i])])
            B1 = B2
        return B1
            
    def affineMap(self, W, b = np.array([])):
        """
            Performs affine mapping of a Box: B = W * x + b

            W : affine mapping matrix, 2D numpy array
            b : affine mapping vector, 1D numpy array
            return -> a new Box
        """
        assert isinstance(W, np.ndarray), 'error: Weight matrix is not a numpy ndarray'
        assert isinstance(b, np.ndarray), 'error: Bias vector is not a numpy ndarray'
        assert len(W.shape) == 2, 'error: Weight matrix should be 2D numpy array'
        assert W.shape[1] == self.generators.shape[0], 'error: Inconsistent dimension between weight matrix and the Box'

        if b.size:
            assert len(b.shape) == 1, 'error: Bias vector should be 1D numpy array'
            assert W.shape[0] == b.shape[0], 'error: Inconsistent dimension between weight matrix and bias vector'
            new_center = W @ self.center + b
            new_generators = W @ self.generators
        else:
            new_center = W @ self.center
            new_generators = W @ self.generators

        n = new_center.shape[0]
        new_lb = np.zeros(n)
        new_ub = np.zeros(n)

        for i in range(n):
            v = new_generators[i, :].reshape(-1, 1)
            new_lb[i] = new_center[i] - np.linalg.norm(v, 1)
            new_ub[i] = new_center[i] + np.linalg.norm(v, 1)

        return Box(new_lb, new_ub)

    def toStar(self):
        """
            Converts current Box to Star set
            
            return -> created Star
        """
        Z = self.toZono()
        return Z.toStar()

    def toZono(self):
        """
            Converts current Box to Zono (zonotope) set
            
            return -> created Zono
        """
        from zono import Zono
        return Zono(self.center, self.generators)
    
    def toPolytope(self):
        """
            Converts current Box to Polytope set
            
            return -> created Polytope
            reference: Polytope is based on polytope library in Python
                       https://tulip-control.github.io/polytope/
        """
        I = np.eye(self.dim)
        A = np.vstack([I, -I])
        b = np.hstack([self.ub, -self.lb])
        return pc.Polytope(A, b)

    def getRanges(self):
        """
            Gets ranges of a state vector of a Box

            return: np.array([
                        lb : float -> lower bound (1D numpy array)
                        ub : float -> upper bound (1D numpy array)
                    ])
        """
        return np.array([self.lb, self.ub])
    
    def getVertices(self):
        """
            Gets all vertices of the box
            
            return: all vertices of the box
        """
        n = self.lb.shape[0]
        N = 2**n
        V = np.array([])
        for i in range(N):
            b = bin(i)[2:].zfill(n+3)[::-1]
            v = np.zeros(n)
            for j in range(n):
                if b[j] == '1':
                    v[j] = self.ub[j]
                else:
                    v[j] = self.lb[j]
            V = np.vstack([V, v]) if V.size else v
        # delete duplicate vertices
        return np.unique(V, axis=0).T

    def boxHull(boxes):
        """
            Merges boxes into a single box
            boxes : array of boxes
            
            return : Box
        """    
        n = len(boxes)
        for i in range(n - 1):
            assert boxes[i].dim == boxes[i+1].dim, 'error: Inconsistent dimension between boxes'
        
        lb = np.empty([0, boxes[0].dim])
        ub = np.empty([0, boxes[0].dim])
        for i in range(n):
            lb = np.vstack([lb, boxes[i].lb])
            ub = np.vstack([ub, boxes[i].ub])
            
        lb = np.amin(lb, axis = 0)
        ub = np.amax(ub, axis = 0)
        return Box(lb, ub)
    
    def plot(self, color=""):
        """
            Plots a Box using Polytope package
            color : color of Polytope 
                (if color is not provided, then polytope is plotted in random color)
        """
        assert self.dim <= 2 and self.dim > 0, 'error: only 2D box can be plotted'
        
        P = self.toPolytope()
        if color: ax = P.plot(color=color)
        else:     ax = P.plot()
        ranges = np.vstack([self.lb, self.ub]).T.reshape(self.dim**2)
        ax.axis(ranges.tolist())
        plt.show()
        
    def __str__(self):
        print('class: %s' % self.__class__)
        print('lb: [shape: %s | type: %s]' % (self.lb.shape, self.lb.dtype))
        print('ub: [shape: %s | type: %s]' % (self.ub.shape, self.ub.dtype))
        print('dim: %s' % self.dim)
        print('center: [shape: %s | type: %s]' % (self.center.shape, self.center.dtype))
        return 'generators: [shape: %s | type: %s]\n' % (self.generators.shape, self.generators.dtype)

    def __repr__(self):
        return "class: %s \nlb: \n%s \nub: \n%s \ndim:%s \ncenter: \n%s \ngenerators: \n%s\n" % (self.__class__, self.lb, self.ub, self.dim, self.center, self.generators)
