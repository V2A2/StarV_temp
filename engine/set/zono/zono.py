#!/usr/bin/python3
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

import sys, copy

sys.path.insert(0, "engine/set/box")
sys.path.insert(0, "engine/set/star")
from box import *
from star import *

class Zono:
    """
        Class for representing a convex set using Zono (zonotope) set
        author: Sung Woo Choi
        date: 9/26/2021
        
        Representation of a Zono
        ====================================================================
        Zono set Z is defined by:
        Z = (c , <v1, v2, ..., vn>) = c + a1 * v1 + ... + an * vn, 
        where -1 <= ai <= 1,
        c is center, vi is generator
        ====================================================================
    """

    def __init__(self, c, V):
        """
            Constructor using 1D representation of an Box

            c : center vector, 1D numpy array
            V : generator matrix, 2D numpy array
        """
        assert isinstance(c, np.ndarray), 'error: Center vector is not a numpy ndarray'
        assert isinstance(V, np.ndarray), 'error: Generator matrix is not a numpy ndarray'
        assert c.shape[0] == V.shape[0], 'error: Inconsistent dimension between a center vector and a generator matrix'
        assert len(V.shape) == 2, 'error: Generator matrix should be 2D numpy array'
        assert len(c.shape) == 1, 'error: Center vector should be 1D numpy array'

        [self.c, self.V] = copy.deepcopy([c, V])
        self.dim = V.shape[0]

    def affineMap(self, W, b = np.array([])):
        """
            Performs affine mapping of a Zono: Z = W * z + b

            W : affine mapping matrix, 2D numpy array
            b : affine mapping vector, 1D numpy array
            return -> a new Zono
        """
        assert isinstance(W, np.ndarray), 'error: Weight matrix is not a numpy ndarray'
        assert isinstance(b, np.ndarray), 'error: Bias vector is not a numpy ndarray'
        assert W.shape[1] == self.dim, 'error: Inconsistent dimension between weight matrix with the zonotope dimension'
        assert len(W.shape) == 2, 'error: Weight matrix should be 2D numpy array'
        
        if b.size:
            assert len(b.shape) == 1, 'error: Bias vector should be 1D numpy array'
            assert W.shape[0] == b.shape[0], 'error: Inconsistency between weight matrix and bias vector'           
            new_c = W @ self.c + b
            new_V = W @ self.V
        else:
            new_c = W @ self.c
            new_V = W @ self.V

        return Zono(new_c, new_V)
    
    def MinkowskiSum(self, X):
        """
            Performs Minkowski Sum of two zonotopes
            X : input Zono (zonotope)

            return -> a new Zono
        """        
        assert isinstance(X, Zono), 'error: Input set, X, is not a zonotope'
        assert self.dim == X.dim, 'error: Inconsistent dimension of input zonotope and this zonotope'
        
        return Zono(self.c + X.c, np.hstack([self.V, X.V]))
    
    def convexHull(self, X):
        """
            Performs convex hull with another zonotope.
            Generally not a zonotope, this function return an over-approximation (a zonotope) of the convex hull.
            
            reference:
                1) Reachability of Uncertain Linear Systems Using Zonotopes, Antoin Girard, HSCC2005.

            X : input Zono (zonotope)
            
            return -> a new Zono
        
            =====================================================================
            Convex hull of two zonotopes is generally NOT a zonotope.
            This method returns an over-approximation (which is a zonotope)
            of a convex hull of two zonotopes.
            This method is a generalization of the method proposed in reference 1.
            In reference 1: the author deals with: CONVEX_HULL(Z, L*Z)
            Here we deals with a more general case: CONVEX_HULL(Z1, Z2)
            We will see that if Z2 = L * Z1, the result is reduced to the result
            obtained by the reference 1.
            We define CH as a convex hull operator and U is union operator.
            Z1 = (c1, <g1, g2, ..., gp>), Z2 = (c2, <h1, ...., hq>)
            =====================================================================
            CH(Z1 U Z2) := {a * x1 + (1 - a) * x2}| x1 \in Z1, x2 \in Z2, 0 <= a <= 1}
            Let a = (e + 1)/2, -1<= e <=1, we have:
                CH(Z1 U Z2) := {(x1 + x2)/2 + e * (x1 - x2)/2}
                            = (Z1 + Z2)/2 + e*(Z1 + (-Z2))/2
                            where, '+' denote minkowski sum of two zonotopes
            From minkowski sum method, one can see that:
                CH(Z1 U Z2) = 0.5*(c1 + c2 <2g1, ..., 2gp, 2h1, ...2hq, c1 - c2>)
            So, the zonotope that over-approximate the convex hull of two zonotop
            has (p + q) + 1 generators.
                                                        
            So, the zonotope that over-approximate the convex hull of two zonotop
            has (p + q) + 1 generators.
            Let consider the specific case Z2 = L * Z1.
            In this case we have:
                (Z1 + L*Z1)/2 = 0.5 * (I+L) * (c1, <g1, g2, ..., gp>)
                (Z1 - L*Z1)/2 = 0.5 * (I-L) * (c1, <g1, ..., gp>)
                ea * (Z1 - L*Z1)/2 = 0.5*(I-L)*(0, <c1, g1, ..., gp>)
                CH(Z1 U L * Z1) = 0.5*((I + L)*c1, <(I+L)*g1, ..., (I+L)*gp,
                                (I-L)*c1, (I-L)*g1, ..., (I-L)*gp>)
            where I is an identity matrix.
            So the resulted zonotope has 2p + 1 generators.
            =====================================================================
        """       
        assert isinstance(X, Zono), 'error: Input set, X, is not a zonotope'
        assert self.dim == X.dim, 'error: Inconsistent dimension of input zonotope and this zonotope'
        
        new_c = 0.5 * (self.c + X.c)
        new_V = np.hstack([self.V, X.V, 0.5*(self.c - X.c).reshape(self.dim, 1)])
        return Zono(new_c, new_V)
    
    def convexHull_with_linearTransform(self, L):
        """
            Performs convex hull of a zonotope with its linear transformation 

            L : linear transformation matrix
            
            return -> a new Zono with order of n_max/dim
        """
        assert len(L.shape) == 2, 'error: Transformation matrix should be 2D numpy array'
        
        [nL, mL] = L.shape
        
        assert nL == mL, 'error: Trasformation matrix should be a square matrix'
        assert nL == self.dim, 'error: Inconsistent dimension of tranformation matrix and this zonotope'
        
        M1 = np.eye(nL) + L
        M2 = np.eye(nL) - L
        
        new_c = 0.5 * M1 @ self.c
        new_V = 0.5 * np.hstack([M1 @ self.V, M2 @ self.c.reshape(self.dim, 1), M2 @ self.V])
        return Zono(new_c, new_V) 
    
    def orderReduction_box(self, n_max):
        """
            Performs order reduction of a zonotope
            
            references:
                1) Reachability of Uncertain Linear Systems Using Zonotopes, Antoine Girard, HSCC 2008 (main reference)
                2) Methods for order reduction of Zonotopes, Matthias Althoff, CDC 2017
                3) A State bounding observer based on zonotopes, C. Combastel, ECC 2003
            
            n_max : maximum allowable number of generators
            
            return -> a new Zono
        
        ======================================================================================
                    ZONOTOPE ORDER REDUCTION USING BOX METHOD
        
        We define: R(Z) is the reduced-order zonotope that over-approximates
        zonotope Z, i.e., Z \subset R(Z).
        
        IDEA: R(Z) is generated by less segments than Z.
        
        PRINCIPLE OF REDUCTION: "The edge of Z with lower length having priority
        to be involved in the reduction". Read references 1, 3, 4, 5 for more information.
        
        STEPS:
           0) Z = (c, <g1, g2, ..., gp>) = c + Z0, Z0 = (0, <g1, g2, ..., gp>)
              we do order reduction for zonotope Z0, and then shift it to a new center point c.
        
           1) Sort the generators g1, g2, ..., gp by their length to have:
              ||g1|| <= ||g2|| <= ||g3||.... <= ||gp||, where ||.|| is the 2-norm.
              This is the heuristic used in reference 3.
              we can use another heuristic used in reference 1 that is:
                 ||g1||_1 - ||g1||_\inf <= .... <= ||gp||_1 - ||gp||_\inf.
              where ||.||_1 is 1-norm and ||.||_\inf is the infinity norm.
        
           2) cur_order = p/n is the current order in which p is the number of generators,
              n is the dimension. r is the desired order of the reduced zonotope R(Z).
              r is usually selected as 1.
              Let d = cur_order - r be the number of orders needed to be reduced.
              We have p = (r + d)*n = (r-1)*n + (d+1)*n
        
           3) The zonotope Z0 is splited into 2 zonotopes: Z0 = Z01 + Z02
              Z01 contains (d + 1)*n smallest generators
              Z02 contains (r - 1)*n lagest generators
        
           4) Over-approximate Z01 by an interval hull IH(Z01), see references 3, 4, 5
              The IH(Z01) has n generators
        
           5) The reduced zonotope is the Minkowski sum of IH(Z01) and Z02:
                           R(Z0) = IH(Z01) + Z02
              R(Z0) has (r-1)*n + n = r*n generators. So it has order of r.
        ======================================================================================
        """
        assert n_max >= self.dim, 'error: n_max should be >= %d' % self.dim
        
        # number of generators
        n = self.V.shape[1] 
        
        if n <= n_max:
            return self    
        else:
            if n_max > self.dim:
                # number of generators need to be reduced
                n1 = n - n_max + self.dim
                
                # sort generatros based on their lengths
                length_gens = np.zeros(n)
                for i in range(n):
                    length_gens[i] = np.linalg.norm(self.V[:, i], 2)
                sorted_ind = length_gens.argsort()

                sorted_gens = np.zeros([self.dim, n])
                for i in range(n):
                    sorted_gens[:, i] = self.V[:, sorted_ind[i]]

                
                Z1 = Zono(np.zeros(self.dim), sorted_gens[:, 0:n1])
                Z2 = Zono(self.c, sorted_gens[:, n1:n])
                return Z2.MinkowskiSum(Z1.getIntervalHull())
                    
            if n_max == self.dim:
                return self.getIntervalHull()
            
    
    def getBox(self):
        """
            Gets a box bound of the zonotope
            
            return -> created Box
        """
        from box import Box
        
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)
        for i in range(self.dim):
            lb[i] = self.c[i] - np.linalg.norm(self.V[i, :], 1)
            ub[i] = self.c[i] + np.linalg.norm(self.V[i, :], 1)
        return Box(lb, ub)
    
    
    def toPolytope(self):
        """
            Converts current Zono to Polytope set
            
            return -> created Polytope
            reference: Polytope is based on polytope library in Python
                       https://tulip-control.github.io/polytope/
        """
        I = np.eye(self.dim)
        C = np.vstack([I, -I])
        d = np.ones(self.dim*2)

        # suppose polytope P = {x element of P : C x <= d} 
        # then, new_P = V * P + c
        V_inv = np.linalg.pinv(self.V)
        new_C = np.dot(C, V_inv)
        new_d = d + np.dot(new_C, self.c)
        return pc.Polytope(new_C, new_d)

    def toStar(self):
        """
            Converts current Zono to Star set

            return -> created Star
        """
        from star import Star
        
        n = self.V.shape[1]
        lb = -np.ones(n)
        ub = np.ones(n)

        C = np.vstack([np.eye(n), -np.eye(n)])
        d = np.hstack([np.ones(n), np.ones(n)])
        V = np.hstack([self.c.reshape(-1, 1), self.V])
        return Star(V, C, d, lb, ub, self)

    def toImageZono(obj, height, width, numChannels):
        """
            Converts Zono to ImageZono
            height: height of the image
            width: width of the image
            numChannels: number of channels of the image
            
            return -> ImageZono
        """
        from imagezono import ImageZono

        assert height*width*numChannels == obj.dim, 'error: Inconsistent dimension, please change the height, width and numChannels to be consistent with the dimension of the zonotope' 

        new_V = np.hstack([obj.c.reshape(-1,1), obj.V])
        numPreds = obj.V.shape[1]
        new_V = new_V.reshape([height, width, numChannels, numPreds+1])
        return ImageZono(new_V)

    # convert to ImageStar
    def toImageStar(obj, height, width, numChannels):
        # @height: height of the image
        # @width: width of the image
        # @numChannels: number of channels of the image
        # return: ImageStar
        im1 = obj.toStar()
        return im1.toImageStar(numChannels, height, width)
    
    def intersectHalfSpace(self, H, g):
        """
            Computes intersection of a Zono with a half space:
            H(x) := Hx <= g
            
            H : halfspace matrix
            g : halfspace vector
            
            return -> a new Star that contain an intersection of a zonotope and halfspace
        """
        S = self.toStar()
        return S.intersectHalfSpace(H, g) 
    
#------------------check if this function is working--------------------------------------------
    def getMaxIndexes(self):
        """
            Returns possible max indexes
            
            return:
                max_ids -> index of the state that can be a max point
        """
        new_rs = self.toStar()
        new_rs = new_rs.toImageStar(self.dim, 1, 1)
        max_id = new_rs.get_localMax_index(np.array([1, 1], np.array(self.dim, 1), 1))

    def contains(self, x):
        """
            Checks if a zonotope contains a point
            x : a point (1D numpy array)
            
            return:
                True -> if the zonotope contain a point x
                False -> if the zonotope does not contain a point x
        """
        assert isinstance(x, np.ndarray), 'error: x is not a numpy array (numpy.ndarray)'
        assert len(x.shape) == 1, 'error: Invalid input point, X should be 1D numpy array'
        assert x.shape[0] == self.dim, 'error: Inconsistent dimension between the input point and the zonotope'
        
        d = x - self.c
        abs_V = abs(self.V)
        d1 = np.sum(abs_V, axis=1)
        
        x1 = (d <= d1)
        x2 = (d >= -d1)
        
        return (sum(x1) == self.dim) and (sum(x2) == self.dim)
    
    def getBounds(self):
        """
            Clip method from Stanley Bak to get bounds of a Zono

            return: np.array([
                        lb : float -> lower bound (1D numpy array)
                        ub : float -> upper bound (1D numpy array)
                    ])
        """
        pos_mat = self.V.transpose()
        neg_mat = self.V.transpose()
        pos_mat = np.where(neg_mat > 0, neg_mat, 0)
        neg_mat = np.where(neg_mat < 0, neg_mat, 0)
        pos1_mat = np.ones(self.V.shape[1])
        ub = pos1_mat @ (pos_mat - neg_mat)
        lb = -ub
        return [self.c + lb, self.c + ub]

    def getRanges(self):
        """
            Getg ranges of a Zono

            return: np.array([
                        lb -> lower bound (1D numpy array)
                        ub -> upper bound (1D numpy array)
                    ])
        """
        B = self.getBox()
        return B.getRanges()

    def getRange(obj, index):
        """
            Get range of a zonotope at specific index

            index : index of the state x[index] 

            return: 
                np.array([
                    lb -> lower bound of x[index] (1D numpy array)
                    ub -> upper bound of x[index] (1D numpy array)
                ])
        """
        if index < 0 or index >= obj.dim:
            raise Exception('error: Invalid idnex')
        
        lb = obj.c[index] - np.linalg.norm(obj.V[index, :], 1)
        ub = obj.c[index] + np.linalg.norm(obj.V[index, :], 1)
        return np.array([lb, ub])

    def is_p1_larger_than_p2(self, p1_id, p2_id):
        """
            Checks if an index of a point in Zono is larger than an index of other point.
            This function is based on Star.is_p1_larger_than_p2() function.
        
            p1_id : index of point 1
            p2_id : index of point 2
            
            return:
                True -> if there exists the case that p1 >= p2
                False -> if there is no case that p1 >= p2
        """
        S = self.toStar()
        return S.is_p1_larger_than_p2(p1_id, p2_id)
    
    def getOrientedBox(self):
        """
            Gets an oriented rectangular hull enclosing a zonotope
            
            return -> created Zono
             
            reference: MATTISE of Prof. Girard, in 2005
        """
        U, _, _ = np.linalg.svd(self.V)
        P = U.T @ self.V
        D = np.diag(np.sum(abs(P), axis=1))
        return Zono(self.c, U @ D)
        
    def getIntervalHull(self):
        """
            Performs interval hull of zonotope

            return -> created Zono
        """
        B = self.getBox()
        return B.toZono()

    def getSupInfinityNorm(self):
        """
            Gets sup_{x\in Z }||x||_\infinity
            
            return -> sup infinity norm
        """
        B = self.getBox()
        V1 = np.hstack([abs(B.lb), abs(B.ub)])
        return np.max(V1)
        
    def getVertices(self):
        """
            Gets all vertices of a zonotope
            
            return: all vertices of the zonotope
        """
        from box import Box
        # number of generators
        n = self.V.shape[1]
        lb = -np.ones(n)
        ub = np.ones(n)
        B = Box(lb, ub)
        V1 = B.getVertices()
        # number of vertices of the zonotope
        m = V1.shape[1]
        V = np.array([])
        for i in range(m):
            v = self.c + self.V @ V1[:, i]
            V = np.vstack([V, v]) if V.size else v
        return V.T

    def plot(self, color=""):
        """
            Plots a Zono using Polytope package
            color : color of Polytope 
                (if color is not provided, then polytope is plotted in random color)
        """
        assert self.dim <= 2 and self.dim > 0, 'error: only 2D zono can be plotted'
        
        P = self.toPolytope()
        if color: ax = P.plot(color=color)
        else:     ax = P.plot()
        [lb, ub] = self.getRanges()
        ranges = np.vstack([lb, ub]).T.reshape(self.dim**2)
        ax.axis(ranges.tolist())
        plt.show()

    def __str__(self):
        print('class: %s' % self.__class__)
        print('c: [shape: %s | type: %s]' % (self.c.shape, self.c.dtype))
        print('V: [shape: %s | type: %s]' % (self.V.shape, self.V.dtype))
        return "dim: %s\n" % (self.dim)

    def __repr__(self):
        return "class: %s \nc: %s \nV: %s \ndim: %s \n" % (self.__class__, self.c, self.V, self.dim)