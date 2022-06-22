#!/usr/bin/python3
import numpy as np
import scipy
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB 
import polytope as pc

# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as a3
# import matplotlib.colors as colors

import sys, copy

sys.path.insert(0, "engine/set/imagestar")
sys.path.insert(0, "engine/set/zono")
sys.path.insert(0, "engine/set/box")
from imagestar import *
from zono import *
from box import *

class Star:
    """
        Class for representing a convex set using Star set
        author: Sung Woo Choi
        date: 9/21/2021
        
        Representation of a Star
        ====================================================================
        Star set defined by 
        x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
            = V * b,
        where V = [c v[1] v[2] ... v[n]],
                b = [1 a[1] a[2] ... a[n]]^T,
                C*a <= d, constraints on a[i]
        ====================================================================
    """

    def __init__(self, *args):
        """
            Star is a tuple consisting <C, V, d>.
            V is a basic matrix (2D numpy array) 
            C is a constraint vector (2D numpy array)
            d is a constraint vector (1D numpy array)
            dim is dimension of star set (non-negative number)
            nVar is number of variables in the constraints (non-negative number)
            predicate_lb is lower bound vector of predicate variables (1D numpy array)
            predicate_ub is upper bound vector of predicate variables (1D numpy array)
            state_lb is lower bound vector of state variables (1D numpy array)
            state_ub is upper bound vector of state variables (1D numpy array)
            Z is an outer Zonotope convering the current star. It is used for reachability of logsig and tansig networks (Zono set)
        """
        from box import Box
        from zono import Zono
        
        # Initialize Star properties with empty numpy sets and zero values.
        [self.V, self.C, self.d] = [np.array([]), np.array([]), np.array([])]
        [self.dim, self.nVar] = [0, 0]
        [self.predicate_lb, self.predicate_ub] = [np.array([]), np.array([])]
        [self.state_lb, self.state_ub] = [np.array([]), np.array([])]
        self.Z = np.array([])

        length_args = len(args)
        if length_args != 6 and length_args != 1:
            args = [element.astype('float64') for element in args]
            
        if length_args == 7:
            [V, C, d, pred_lb, pred_ub, state_lb, state_ub] = copy.deepcopy(args)
            self.check_essential_properties(V, C, d)
            self.check_predicate_bounds(pred_lb, pred_ub, C, d)
            self.check_state_bounds(state_lb, state_ub, V)
            
        elif length_args == 6:
            [V, C, d, pred_lb, pred_ub, outer_zono] = copy.deepcopy(args)
            self.check_essential_properties(V, C, d)
            self.check_predicate_bounds(pred_lb, pred_ub, C, d)
            self.check_outer_zono(outer_zono, V)
            
        elif length_args == 5:
            [V, C, d, pred_lb, pred_ub] = copy.deepcopy(args)
            self.check_essential_properties(V, C, d)
            self.check_predicate_bounds(pred_lb, pred_ub, C, d)
            
        elif length_args == 3:
            [V, C, d] = copy.deepcopy(args)
            self.check_essential_properties(V, C, d)
            
        elif length_args == 2:
            [lb, ub] = copy.deepcopy(args)
            assert lb.shape == ub.shape, 'error: Inconsistent dimension between upper- and lower- bound vectors'
            assert len(lb.shape) == len(ub.shape) == 1, 'error: Lower- and upper-bound vectors should be 1D numpy array'
            
            B = Box(lb, ub)
            S = B.toStar()
            self.V = S.V
            self.C = np.zeros([1, S.nVar])
            self.d = np.zeros(1)
            self.dim = S.dim
            self.nVar = S.nVar
            self.state_lb = lb
            self.state_ub = ub
            
            self.predicate_lb = -np.ones(S.nVar)
            self.predicate_ub = np.ones(S.nVar)
            self.Z = B.toZono()
            
        elif length_args == 1:
            from polytope import Polytope
            from box import Box
            
            [P] = copy.deepcopy(args)
            assert isinstance(P, Polytope), 'error: Input set is not a Polytope'
            
            c = np.zeros([P.dim, 1])
            I = np.eye(P.dim)
            V = np.hstack([c, I])
            
            self.check_essential_properties(np.hstack([c, I]), P.A, P.b)
            
            self.predicate_lb, self.predicate_ub = self.getRanges()
            B = Box(self.predicate_lb, self.predicate_ub)
            self.Z = B.toZono()
            
        elif length_args == 0:
            # create empty Star (for preallocation an array of Star)
            pass
        
        else:
            raise Exception('error: Invalid number of input arguments (should be 0, 2, 3, 5, 6, or 7)')
        
    def check_essential_properties(self, V, C, d):
        """
            Checks essential properties of star (V, C, d) before insearting to Star
            
            return:
                V -> a basic matrix (2D numpy array) 
                C -> a constraint vector (2D numpy array)
                d -> a constraint vector (1D numpy array)
        """
        assert len(C.shape) == 2, 'error: Constraint matrix should be 2D numpy array'
        assert len(d.shape) == 1, 'error: Constraint vector should be 1D numpy array'
        
        [nV, mV] = V.shape[0:2]
        [nC, mC] = C.shape
        [nd] = d.shape
        assert mV == mC + 1, 'error: Inconsistency between basic matrix and constraint matrix'
        assert nC == nd, 'error: Inconsistency between constraint matrix and constraint vector'
        
        [self.V, self.C, self.d] = [V, C, d]
        self.dim = nV
        self.nVar = mC
    
    def check_predicate_bounds(self, pred_lb, pred_ub, C, d):
        """
            Checks predicate bounds before insearting into Star
        """
        if pred_lb.size > 0 and pred_ub.size > 0:
            assert len(pred_ub.shape) == len(pred_lb.shape) == 1, \
                'error: Predicate lower- and upper-bound vectors should be 1D numpy array'

            [nC, mC] = C.shape
            [nd, n1, n2] = [d.shape[0], pred_lb.shape[0], pred_ub.shape[0]]          
            assert n1 == n2 and n1 == mC, 'error: Inconsistency between number of predicate variables and predicate lower- or upper-bound vectors'

        [self.predicate_lb, self.predicate_ub] = [pred_lb, pred_ub]
    
    def check_state_bounds(self, state_lb, state_ub, V):
        """
            Checks states bounds before insearting into Star
        """
        if state_lb.size > 0 and state_ub.size > 0:
            assert state_lb.size > 0 and state_ub.size > 0 and len(state_ub.shape) == len(state_lb.shape) == 1, \
                    'error: State lower- and upper-bound vectors should be 1D numpy array'
            [nV, mV] = V.shape
            [n1, n2] = [state_lb.shape[0], state_ub.shape[0]]  
            assert n1 == n2 and n1 == nV, 'error: Inconsistent dimension between lower- and upper- bound vectors of state variables and matrix V'
        
        [self.state_lb, self.state_ub] = [state_lb, state_ub]
        
    def check_outer_zono(self, outer_zono, V):
        """
            Checks outer zonotope before insearting into Star
        """  
        from zono import Zono
        if isinstance(outer_zono, Zono):
            assert outer_zono.V.shape[0] == V.shape[0], 'error: Inconsistent dimension between outer zonotope and star set'
            self.Z = outer_zono
        else:
            assert outer_zono.size == 0, 'error: Outer zonotope is not a Zono class'
            
    def isEmptySet(self):
        """
            Checks if Star set is an empty set.
            A Star set is an empty if and only if the predicate P(a) is infeasible.
            
            return:
                2 -> True; star is an empty set
                3 -> False; star is a feasible set
                else -> error code from Gurobi LP solver
        """
        # error code (m.status) description avaliable at
        # https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
        # parameter settings: https://www.gurobi.com/documentation/9.1/refman/parameters.html
        
        if not (self.V.size and self.C.size and self.d.size):
            return 2
        
        f = np.zeros(self.nVar)
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if self.predicate_lb.size and self.predicate_ub.size:
            x = m.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
        else:
            x = m.addMVar(shape=self.nVar)
        m.setObjective(f @ x, GRB.MINIMIZE)
        A = sp.csr_matrix(self.C)
        b = self.d
        m.addConstr(A @ x <= b)     
        m.optimize()

        if m.status == 2:
            return False
        elif m.status == 3:
            return True
        else:
            raise Exception('error: exitflat = %d' % (m.status))

    def contains(self, s):
        """
            Checks if a Star set contains a point.
            s : a star point (1D numpy array)
            
            return :
                1 -> a star set contains a point, s 
                0 -> a star set does not contain a point, s
                else -> error code from Gurobi LP solver
        """
        assert len(s.shape) == 1, 'error: Invalid star point. It should be 1D numpy array'
        assert s.shape[0] == self.dim, 'error: Dimension mismatch'     
        
        f = np.zeros(self.nVar)
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if self.predicate_lb.size and self.predicate_ub.size:
            x = m.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
        else:
            x = m.addMVar(shape=self.nVar)
        m.setObjective(f @ x, GRB.MINIMIZE)
        C = sp.csr_matrix(self.C)
        m.addConstr(C @ x <= self.d)
        Ae = sp.csr_matrix(self.V[:, 1 : self.nVar + 1])
        be = s - self.V[:, 0]
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))
        
    def sample(self, N):
        """
            Samples number of points in the feasible Star set 
            N : number of points in the sample
            
            return :
                V -> a set of at most N sampled points in the star set 
        """
        from box import Box
        assert N >= 1, 'error: Invalid number of samples'

        B = self.getBox()
        if not isinstance(B, Box):
            V = np.array([])
        else:
            [lb, ub] = B.getRanges()

            V1 = np.array([])
            for i in range(self.dim):
                X = (ub[i] - lb[i]) * np.random.rand(2*N, 1) + lb[i]
                V1 = np.hstack([V1, X]) if V1.size else X

            V = np.array([]).reshape(0, self.dim)
            for i in range(2*N):
                v1 = V1[i,:]
                if self.contains(v1):
                    V = np.vstack([V, v1])

            V = V.T
            if V.shape[1] >= N:
                V = V[:, 0:N]

        return V

    def affineMap(self, W, b = np.array([])):
        """
            Performs affine mapping of a Star: S = W * x + b

            W : affine mapping matrix, 2D numpy array
            b : affine mapping vector, 1D numpy array
            return -> a new Star
        """
        assert isinstance(W, np.ndarray), 'error: Weight matrix is not a numpy ndarray'
        assert isinstance(b, np.ndarray), 'error: Bias vector is not a numpy ndarray'
        assert W.shape[1] == self.dim, 'error: Inconsistent dimension between weight matrix with the Star dimension'
        assert len(W.shape) == 2, 'error: Weight matrix should be 2D numpy array'

        if b.size:
            assert len(b.shape) == 1, 'error: Bias vector should be 1D numpy array'
            assert W.shape[0] == b.shape[0], 'error: Inconsistency between weight matrix and bias vector'
            new_V = W @ self.V
            new_V[:, 0] += b
        else:
            new_V = W @ self.V

        if self.Z:
            from zono import Zono
            if isinstance(self.Z, Zono):
                new_Z = self.Z.affineMap(W, b)
            else:
                raise Exception('error: Outter zono is not Zono set')
        else:
            new_Z = np.array([])

        return Star(new_V, self.C, self.d, self.predicate_lb, self.predicate_ub, new_Z)

    def MinkowskiSum(self, X):
        """
            Operates Minkowski Sum
            X : another Star with smake dimension
            
            return -> new Star = (self (+) X), where (+) is Minkowski Sum
        """
        assert self.dim == X.dim, 'error: Inconsistent dimension between current Star and input Star (X)'
        
        V1 = self.V[:, 1:]
        V2 = X.V[:, 1:]
        new_c = (self.V[:, 0] + X.V[:, 0]).reshape(-1, 1)
        
        # check if two Star have the same number of constraints
        if self.C.shape == X.C.shape and np.linalg.norm(self.C - X.C) + np.linalg.norm(self.d - X.d) < 0.0001 :
            V3 = V1 + V2
            new_V = np.hstack([new_c, V3])
            return Star(new_V, self.C, self.d)
        
        else:
            V3 = np.hstack([V1, V2])
            new_V = np.hstack([new_c, V3])
            new_C = scipy.linalg.block_diag(self.C, X.C)
            new_d = np.hstack([self.d, X.d])
            return Star(new_V, new_C, new_d)
        
    def Sum(self, X):
        """
            Operates new Minkowski Sum (used for Recurrent Layer reachability)
            X : another star with same dimension
            
            return -> new Star = (self (+) X), where (+) is Minkowski Sum
        """
        assert isinstance(X, Star), 'error: Input variable X is not a Star'

        V1 = self.V[:, 1:]
        V2 = X.V[:, 1:]

        V3 = np.hstack([V1, V2])
        new_c = (self.V[:, 0] + X.V[:, 0]).reshape(-1, 1)
        new_V = np.hstack([new_c, V3])
        new_C = scipy.linalg.block_diag(self.C, X.C)
        new_d = np.hstack([self.d, X.d])

        if self.predicate_lb.size and X.predicate_lb.size:
            new_predicate_lb = np.hstack([self.predicate_lb, X.predicate_lb])
            new_predicate_ub = np.hstack([self.predicate_ub, X.predicate_ub])
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)
        return Star(new_V, new_C, new_d)

    def intersectHalfSpace(self, H, g):
        """
            Computes intersection of a Star with a half space:
            H(x) := Hx <= g
            
            H : halfspace matrix
            g : halfspace vector
            
            return -> a new Star set with more constraints
        """
        assert isinstance(H, np.ndarray), 'error: Halfspace matrix is not a numpy ndarray'
        assert isinstance(g, np.ndarray), 'error: Halfspace vector is not a numpy ndarray'
        assert len(g.shape) == 1, 'error: Halfspace vector should be 1D numpy array'
        assert H.shape[0] == g.shape[0], 'error: Inconsistent dimension between halfspace matrix and halfspace vector'
        assert H.shape[1] == self.dim, 'error: Inconsistent dimension between halfspace and star set'
        
        H = H.astype('float64')
        g = g.astype('float64')

        m = self.V.shape[1]
        C1 = H @ self.V[:, 1:m]
        d1 = g - H @ self.V[:, 0]

        new_C = np.vstack([self.C, C1])
        new_d = np.hstack([self.d, d1])

        S = Star(self.V, new_C, new_d, self.predicate_lb, self.predicate_ub)

        if S.isEmptySet():
            S = np.array([])
        return S
    
    def scalarMap(self, alp_max):
        """
            Scalar map of a Star S' = alp * S, 0 <= alp <= alp_max
            =============================================================
            S: x = alp*c + V* alph * a, Ca <= d
            note that:   Ca <= d -> C*alph*a <= alp*a <= alp_max * d
            let: beta = alp * a, we have
            S := x = alp * c + V * beta, C * beta <= alp_max * d,
                                         0 <= alp <= alp_max
            Let g = [beta; alp]
            S = Star(new_V, new_C, new_d), where:
              new_V = [0 c V], new_C = [0 -1; 0 1; 0 C], new_d = [0; alpha_max; alp_max * d]
                  
            S has one more basic vector compared with obj
            =============================================================
            a_max : maximum value of a
            
            Note: we always require that alp >= 0
        """
        new_c = np.zeros([self.dim, 1])
        new_V = np.hstack([self.V, new_c])
        new_C = scipy.linalg.block_diag(self.C, np.array([[-1], [1]]))
        new_d = np.hstack([alp_max*self.d, 0, alp_max])
        return Star(new_V, new_C, new_d)
    
    def convexHull(self, X):
        """
            Convex hull of two Stars
            =============================================================
            This method is similar to the method proposed by Prof. Girard
            for Zonotope. See the following paper:
            1) Reachability of Uncertain Linear Systems Using
            Zonotopes, Antoin Girard, HSCC2005.
            
            CH(S1 U S2) := {l*x1 + (1-l)*x2}, 0 <= l <= 1, x1 \in S1, x2
            \in S2.     := {x2 + l*(x1 - x2)}
            
                        := c2 + a2 * V2 + l(c1 - c2) + l*a1*V1 - l*a2*V2
            let beta1 = l * a1, beta2 = l*a2 -> C1 * beta1 <= l * d1 <=
            d1 and C2 * beta2 <= l * d2 <= d2
            
            CH(S1 U S2) = c2 + a2 * V2 + l * (c1 - c2) + beta1 * V1 -
                                                              beta2 * V2
            
            if S2 = L*S1, i.e., S2 is a projection of S1
            
            CH(S1 U S2) := {(l + (1-l)*L) * x1} = {(l(I - L) + L) * x1}
                        := (l * (I - L) * x1 + L * x1)
            =============================================================
        """
        assert isinstance(X, Star), 'error: Input set, X, is not a Star'
        assert X.dim == self.dim, 'error: Inconsistent dimension between input set and current star'
        
        c1 = self.V[:, 0].reshape(-1,1)
        V1 = self.V[:, 1:]
        c2 = X.V[:, 0].reshape(-1,1)
        V2 = X.V[:, 1:]
        
        new_V = np.hstack([c1-c2, V1, -V2])
        new_C = scipy.linalg.block_diag(self.C, X.C)
        new_d = np.hstack([self.d, X.d])
        
        S1 = Star(new_V, new_C, new_d)
        return X.MinkowskiSum(S1)
    
    def convexHull_with_linearTransform(self, L):
        """
            convex hull of a Star with its linear transformation
            L : linear transformation matrix
            
            return -> a new Star
        """
        assert len(L.shape) == 2 and L.shape[0] == L.shape[1], 'error: Transformation matrix, L, should be a square matrix'
        assert L.shape[0] == self.dim, 'error: Inconsistent dimension of transformation and this Star'
        
        M = np.eye(self.dim) - L
        S1 = self.affineMap(L)
        S2 = self.affineMap(M)
        
        new_V = np.hstack([S1.V, S2.V])
        new_C = scipy.linalg.block_diag(S1.C, np.array([[-1], [1]]), S2.C)
        new_d = np.hstack([S1.d, 0, 1, S2.d])
        return Star(new_V, new_C, new_d)
    
    #------------------------------- Need to Test this function ---------------------------------#
    # def orderReduction_box(self, n_max):
    #     """
    #         Order reudction for Stars (similiar order reduction as zonotope)
    #         See the Zono class for the detail
    #         We reduce the number of basic vectors of a Star
    #         n_max : maximum allowable number of basic vectors
            
    #         return -> a new Star with number of basic vectors = n_max
    #     """
    #     assert n_max >= self.dim, 'error: n_max should be >= %d' % self.dim
        
    #     if n_max == self.dim:
    #         B = self.getBox()
    #         S = B.toStar()
        
    #     # number of generators
    #     n = self.V.shape[1] - 1 
    #     if n_max > self.dim:
    #         # number of generators need to be reduced
    #         n1 = n- n_max + self.dim
            
    #         # sort generators based on their lenghts
    #         length_gens = np.zeros(n)
    #         for i in range(1, n+1):
    #             length_gens[i] = np.linalg.norm(self.V[:, i], 2)
    #         sorted_ind = np.argsort(length_gens)
    #         sorted_gens = np.zeros([self.dim, n])
    #         for i in range(n):
    #             sorted_gens[:, i] = self.V[:, sorted_ind[i]]
    #         return []
        
    def toPolytope(self):
        """
            Converts to Polytope
        """
        b = self.V[:, 0]
        W = self.V[:, 1:]
        
        if self.predicate_lb.size and self.predicate_ub.size:
            I = np.eye(self.nVar)
            C1 = np.vstack([I, -I])
            d1 = np.hstack([self.predicate_ub, -self.predicate_lb])
            
            C = np.vstack([self.C, C1])
            d = np.hstack([self.d, d1])

        else:
            C = self.C
            d = self.d

        # suppose polytope P = {x element of P : C x <= d} 
        # then, new_P = W * P + b
        W_inv = np.linalg.pinv(W)
        new_C = np.dot(C, W_inv)
        new_d = d + np.dot(new_C, b)
        return pc.Polytope(new_C, new_d)

    def toImageStar(self, height, width, numChannel):
        """
            Converts current Star to ImageStar set
            height : height of ImageStar
            width : width of ImageStar
            numChannel : number of channels in ImageStar
            
            return -> ImageStar
        """
        from imagestar import ImageStar

        assert self.dim == height*width*numChannel, 'error: inconsistent dimension in the ImageStar and the original Star set'
        
        new_V = np.reshape(self.V, (height, width, numChannel, self.nVar + 1))
        return ImageStar(new_V, self.C, self.d, self.predicate_lb, self.predicate_ub)

    def getBox(self):
        """
            Finds a box bound of a star set
            
            return -> created Box
        """
        from box import Box

        if self.C.size == 0 or self.d.size == 0 :
            # star set is just a vector (one point)
            lb = self.V[:, 0]
            ub = self.V[:, 0]
            return Box(lb, ub)
        else:
            # star set is a set
            if self.state_lb.size and self.state_ub.size:
                return Box(self.state_lb, self.state_ub)
            else:
                lb = np.zeros(self.dim)
                ub = np.zeros(self.dim)

                for i in range(self.dim):
                    f = self.V[i, 1:self.nVar + 1]
                    if (f == 0).all():
                        lb[i] = self.V[i, 0]
                        ub[i] = self.V[i, 0]
                    else:
                        lb_ = gp.Model()
                        lb_.Params.LogToConsole = 0
                        lb_.Params.OptimalityTol = 1e-9
                        if self.predicate_lb.size and self.predicate_ub.size:
                            x = lb_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                        else:
                            x = lb_.addMVar(shape=self.nVar)
                        lb_.setObjective(f @ x, GRB.MINIMIZE)
                        C = sp.csr_matrix(self.C)
                        lb_.addConstr(C @ x <= self.d)
                        lb_.optimize()

                        if lb_.status == 2:
                            lb[i] = lb_.objVal + self.V[i, 0]
                        else:
                            raise Exception('error: Cannot find an optimal solution, exitflag = %d' % (lb_.status))

                        ub_ = gp.Model()
                        ub_.Params.LogToConsole = 0
                        ub_.Params.OptimalityTol = 1e-9
                        if self.predicate_lb.size and self.predicate_ub.size:
                            x = ub_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                        else:
                            x = ub_.addMVar(shape=self.nVar)
                        ub_.setObjective(f @ x, GRB.MAXIMIZE)
                        C = sp.csr_matrix(self.C)
                        ub_.addConstr(C @ x <= self.d)
                        ub_.optimize()

                        if ub_.status == 2:
                            ub[i] = ub_.objVal + self.V[i, 0]
                        else:
                            raise Exception('error: Cannot find an optimal solution, exitflag = %d' % (ub_.status))

                if lb.size == 0 or ub.size == 0:
                    return np.array([])
                else:
                    return Box(lb, ub)
    
    #------------------------------- Need to Test this function ---------------------------------#
    def getMaxIndexes(self):
        """
            Returns possible max indexes
            
            return -> index of the state that can be a max point 
        """
        new_rs = self.toImageStar(self.dim, 1, 1)
        print('\nPrint converted ImageStar in detail: \n')
        print('numChannel: ', new_rs.get_num_channel())
        print('height: ', new_rs.get_height())
        print('width: ', new_rs.get_width())
        print('V: ', new_rs.get_V())
        print('C: ', new_rs.get_C())
        print('d: ', new_rs.get_d())
        print('pred_lb: ', new_rs.get_pred_lb())
        print('pred_ub: ', new_rs.get_pred_ub())
        max_id = new_rs.get_localMax_index(np.array([0, 0]), np.array([self.dim, 1]), 0)
        return max_id[:, 1]
    
    def getBoxFast(self):
        """
            Estimates ranges of the state vector quickly
            These ranges are not the exact ranges,
            The function finds ranges that are over-approximation of the exact ranges
            
            return -> Box
        """
        from box import Box
        
        if self.C.size == 0 or self.d.size == 0: # star set is just a vector (one point)
            lb = self.V[:, 0]
            ub = self.V[:, 0]
            return Box(lb, ub)
        else:
            [pred_lb, pred_ub] = self.getPredicateBounds()
            B1 = Box(pred_lb, pred_ub)
            Z = B1.toZono()
            Z = Z.affineMap(self.V[:, 1:], self.V[:, 0])
            return Z.getBox()

    def getPredicateBounds(self):
        """
            Gets bounds of predicate variables
            
            return:
                np.array([
                    pred_lb -> predicate lower bound vector (1D numpy array)
                    pred_ub -> predicate upper bound vector (1D numpy array)
                ])
        """
        if self.predicate_lb.size and self.predicate_ub.size:
            return np.array([self.predicate_lb, self.predicate_ub])
        
        else:
            center = np.zeros([self.dim, 1])
            I = np.eye(self.dim)
            V = np.hstack([center, I])
            S = Star(V, self.C, self.d)
            return S.getRanges()

    def getRange(self, index):
        """
            Finds range of a state at specific position
            
            return: 
                np.array([
                    xmin -> min value of x[index]
                    xmax -> max value of x[index]
                ])
        """
        assert isinstance(index, int), 'error: index is not an integer'
        if index < 0 or index >= self.dim:
            raise Exception('error: invalid index')

        f = self.V[index, 1:self.nVar + 1]
        if (f == 0).all():
            xmin = self.V[index, 0]
            xmax = self.V[index, 0]
        else:
            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            if self.predicate_lb.size and self.predicate_ub.size:
                x = min_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
            else:
                x = min_.addMVar(shape=self.nVar)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(self.C)
            min_.addConstr(C @ x <= self.d)
            min_.optimize()

            if min_.status == 2:
                xmin = min_.objVal + self.V[index, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            if self.predicate_lb.size and self.predicate_ub.size:
                x = max_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
            else:
                x = max_.addMVar(shape=self.nVar)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(self.C)
            max_.addConstr(C @ x <= self.d)
            max_.optimize()

            if max_.status == 2:
                xmax = max_.objVal + self.V[index, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))

        return np.array([xmin, xmax])

    def getMin(self, index, lp_solver = 'gurobi'):
        """
            Finds lower bound state variable using LP solver
            index : position of the state
            lp_solver :
                - 'gurobi' : Gurobi linear programming solver
                
            return : 
                xmin -> min value of x[index]
                    
        """
        # assert isinstance(index, int), 'error: index is not an integer'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if index < 0 or index >= self.dim:
            raise Exception('error: invalid index')

        f = self.V[index, 1:self.nVar + 1]
        if (f == 0).all():
            xmin = self.V[index, 0]
        else:
            if lp_solver == 'gurobi':
                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if self.predicate_lb.size and self.predicate_ub.size:
                    x = min_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                else:
                    x = min_.addMVar(shape=self.nVar)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                C = sp.csr_matrix(self.C)
                min_.addConstr(C @ x <= self.d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))
            else:
                raise Exception('error: unknown lp solver, should be gurobi')
        return xmin

    def getMins(self, map, par_option = 'single', dis_option = '', lp_solver = 'gurobi'):
        """
            Finds lower bound vector of state variable using LP solver
            map : an array of indexes
            par_option :
                - 'single' : normal for loop
                - 'parallel' : parallel computation of for loop
            dis_option :
                - 'display' : display the process comments
                - else : no comments
            lp_solver :
                - 'gurobi' : Gurobi linear programming solver
                
            return : 
                xmin : min values of x[map]
                    -> lower bound vector (1D numpy array)
        """

        assert isinstance(map, np.ndarray), 'error: map is not a numpy ndarray'
        assert isinstance(par_option, str), 'error: par_option is not a string'
        assert isinstance(dis_option, str), 'error: dis_option is not a string'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        n = len(map)
        xmin = np.zeros(n)
        if not len(par_option) or par_option == 'single':   # get Mins using single core
            reverseStr = ''
            for i in range(n):
                xmin[i] = self.getMin(map[i], lp_solver)
# implement display option----------------------------------------------------------------------------------------
                # if dis_option = 'display':

                # msg = "%d/%d" % (i, n)
                # print([reverseStr, msg])
                # msg = sprintf('%d/%d', i, n);
                # fprintf([reverseStr, msg]);
                # reverseStr = repmat(sprintf('\b'), 1, length(msg));
# implement multiprogramming for parallel computation of bounds----------------------------------------------------
        elif par_option == 'parallel':  # get Mins using multiple cores
            print('warning: Parallel computation is not implemented yet!!! => computing with normal for loop')
            f = self.V[map, 1:self.nVar + 1]
            V1 = self.V[map, 0]
            for i in range (n):
                if f[i, :].sum() == 0:
                    xmin[i] = V1[i, 0]
                else:
                    if lp_solver == 'gurobi':
                        min_ = gp.Model()
                        min_.Params.LogToConsole = 0
                        min_.Params.OptimalityTol = 1e-9
                        if self.predicate_lb.size and self.predicate_ub.size:
                            x = min_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                        else:
                            x = min_.addMVar(shape=self.nVar)
                        min_.setObjective(f[i, :] @ x, GRB.MINIMIZE)
                        C = sp.csr_matrix(self.C)
                        min_.addConstr(C @ x <= self.d)
                        min_.optimize()

                        if min_.status == 2:
                            xmin[i] = min_.objVal + V1[i]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))
        else:
            raise Exception('error: unknown lp solver, should be gurobi')
        return xmin

    def getMax(self, index, lp_solver = 'gurobi'):
        """
            Finds upper bound state variable using LP solver
            index : position of the state
            lp_solver :
                - 'gurobi' : Gurobi linear programming solver
                
            return : 
                xmax -> max value of x[index]
        """
        # assert isinstance(index, int), 'error: index is not an integer'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if index < 0 or index >= self.dim:
            raise Exception('error: invalid index')

        f = self.V[index, 1:self.nVar + 1]
        if (f == 0).all():
            xmax = self.V[index, 0]
        else:
            if lp_solver == 'gurobi':
                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if self.predicate_lb.size and self.predicate_ub.size:
                    x = max_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                else:
                    x = max_.addMVar(shape=self.nVar)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                C = sp.csr_matrix(self.C)
                max_.addConstr(C @ x <= self.d)
                max_.optimize()
                if max_.status == 2:
                    xmax = max_.objVal + self.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
            else:
                raise Exception('error: unknown lp solver, should be gurobi')
        return xmax

    def getMaxs(self, map, par_option = 'single', dis_option = '', lp_solver = 'gurobi'):
        """
            Finds upper bound vector of state variable using LP solver
            map : an array of indexes
            par_option :
                - 'single' : normal for loop
                - 'parallel' : parallel computation of for loop
            dis_option :
                - 'display' : display the process comments
                - else : no comments
            lp_solver :
                - 'gurobi' : Gurobi linear programming solver
                
            return : 
                xmax : max values of x[map]
                    -> lower bound vector (1D numpy array)
        """
        assert isinstance(map, np.ndarray), 'error: map is not a ndarray'
        assert isinstance(par_option, str), 'error: par_option is not a string'
        assert isinstance(dis_option, str), 'error: dis_option is not a string'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        n = len(map)
        xmax = np.zeros(n)
        if not len(par_option) or par_option == 'single':   # get Mins using single core
            reverseStr = ''
            for i in range(n):
                xmax[i] = self.getMax(map[i], lp_solver)
# implement display option----------------------------------------------------------------------------------------
                # if dis_option = 'display':

                # msg = "%d/%d" % (i, n)
                # print([reverseStr, msg])
                # msg = sprintf('%d/%d', i, n);
                # fprintf([reverseStr, msg]);
                # reverseStr = repmat(sprintf('\b'), 1, length(msg));
# implement multiprogramming for parallel computation of bounds----------------------------------------------------
        elif par_option == 'parallel':  # get Mins using multiple cores
            print('warning: Parallel computation is not implemented yet!!! => computing with normal for loop')
            f = self.V[map, 1:self.nVar + 1]
            V1 = self.V[map, 0]
            for i in range (n):
                if f[i, :].sum() == 0:
                    xmax[i] = V1[i, 0]
                else:
                    if lp_solver == 'gurobi':
                        max_ = gp.Model()
                        max_.Params.LogToConsole = 0
                        max_.Params.OptimalityTol = 1e-9
                        if self.predicate_lb.size and self.predicate_ub.size:
                            x = max_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
                        else:
                            x = max_.addMVar(shape=self.nVar)
                        max_.setObjective(f[i, :] @ x, GRB.MAXIMIZE)
                        C = sp.csr_matrix(self.C)
                        max_.addConstr(C @ x <= self.d)
                        max_.optimize()
                        if max_.status == 2:
                            xmax[i] = max_.objVal + V1[i]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
        else:
            raise Exception('error: unknown lp solver, should be gurobi')
        return xmax
    
    def resetRow(self, map):
        """
            Resets a row of a star set to zero
            map : an array of indexes
            
            return: 
                xmax -> max values of x[indexes] 
        """
        from zono import Zono
        
        V1 = self.V
        V1[map, :] = 0
        if isinstance(self.Z, Zono):
            c2 = self.Z.c
            c2[map] = 0
            V2 = self.Z.V
            V2[map, :] = 0
            new_Z = Zono(c2, V2)
        else:
            new_Z = []
        return Star(V1, self.C, self.d, self.predicate_lb, self.predicate_ub, new_Z)

    def scaleRow(self, map, gamma):
        """
            Scales a row of a star set
            map : an array of indexes
            gamma : scale value
        """
        from zono import Zono
        
        V1 = self.V
        V1[map, :] = gamma*V1[map, :]
        if isinstance(self.Z, Zono):
            c2 = self.Z.c
            c2[map] = gamma*c2[map]
            V2 = self.Z.V
            V2[map, :] = gamma*V2[map, :]
            new_Z = Zono(c2, V2)
        else:
            new_Z = []
        return Star(V1, self.C, self.d, self.predicate_lb, self.predicate_ub, new_Z)

    def getRanges(self):
        """
            Computes lower bound vector and upper bound vector of the state variables
            
            return: 
                np.array([
                    lb -> lower bound vector of x[index]
                    ub -> upper bound vector of x[index]
                ])
        """
        if not self.isEmptySet():
            n = self.dim
            lb = np.zeros(n)
            ub = np.zeros(n)
            for i in range(n):
                [lb[i], ub[i]] = self.getRange(i)
        else:
            lb = np.array([])
            ub = np.array([])
        return np.array([lb, ub])

    def estimateRange(self, index):
        """
            Estimates range of a state variable at specific position
            index : position of the state
            
            return: 
                    np.array([
                        min -> min values of x[index]
                        max -> max values of x[index]
                    ])
        """
        if index < 0 or index >= self.dim:
            raise Exception('error: invalid index')
            
        if self.predicate_lb.size and self.predicate_ub.size:
            f = self.V[index, :]
            xmin = f.item(0)
            xmax = f.item(0)
            for i in range(1, self.nVar+1):
                if f.item(i) >= 0:
                    xmin = xmin + f.item(i) * self.predicate_lb[i-1]
                    xmax = xmax + f.item(i) * self.predicate_ub[i-1]
                else:
                    xmin = xmin + f.item(i) * self.predicate_ub[i-1]
                    xmax = xmax + f.item(i) * self.predicate_lb[i-1]
        else:
            print('The ranges of predicate variables are unknown to estimate the ranges of the states, we solve LP optimization to get the exact range')
            [xmin, xmax] = self.getRange(index)
        return np.array([xmin, xmax])

    def estimateRanges(self):
        """
            Estimates ranges of a state variable using clip method from Stanley Bak
            It is slower than the for-loop method
            
            index : position of the state
            
            return: 
                    np.array([
                        lb -> lower bound vector of x[index]
                        ub -> upper bound vector of x[index]
                    ])
        """
        pos_mat = np.where(self.V > 0, self.V, 0)
        neg_mat = np.where(self.V < 0, self.V, 0)

        xmin1 = pos_mat @ np.hstack([0, self.predicate_lb])
        xmax1 = pos_mat @ np.hstack([0, self.predicate_ub])
        xmin2 = neg_mat @ np.hstack([0, self.predicate_ub])
        xmax2 = neg_mat @ np.hstack([0, self.predicate_lb])
        
        lb = self.V[:, 0] + xmin1 + xmin2
        ub = self.V[:, 0] + xmax1 + xmax2
        return np.array([lb, ub])

    def estimateBound(self, index):
        """
            Estimates lower bound and upper bound vector of state variable at specific index using clip method from Stanely Bak.
            index : position of the state
            
            return : 
                np.array([
                    xmin -> lower bound vector of x[index]
                    xmax -> lower bound vector of x[index]
                ])
        """
        if index < 0 or index >= self.dim:
            raise Exception('error: invalid index')

        f = self.V[index, 1:self.nVar + 1]

        pos_mat = np.array(np.where(f > 0, f, 0))
        neg_mat = np.array(np.where(f < 0, f, 0))

        xmin1 = pos_mat @ self.predicate_lb
        xmax1 = pos_mat @ self.predicate_ub
        xmin2 = neg_mat @ self.predicate_ub
        xmax2 = neg_mat @ self.predicate_lb

        xmin = self.V[index, 0] + xmin1 + xmin2
        xmax = self.V[index, 0] + xmax1 + xmax2
        return np.array([xmin, xmax])

    def estimateBounds(self):
        """
            Quickly estimates lower bound and upper bound vector of state variable
            
            return : 
                np.array([
                    lb -> lower bound vector of x[index]
                    ub -> lower bound vector of x[index]
                ])
        """
        from zono import Zono
        if isinstance(self.Z, Zono):
            [lb, ub] = self.Z.getBounds()
        else:
            n = self.dim
            lb = np.zeros(n)
            ub = np.zeros(n)

            for i in range(n):
                [lb[i], ub[i]] = self.estimateRange(i)
        return np.array([lb, ub])
    
    def get_max_point_candidates(self):
        """
            Estimates quickly max-point candidates
            
            return:
                an array of indexes of max-point candidates
        """
        [lb, ub] = self.estimateRanges()
        id = np.argmax(lb)
        b = (ub >= lb[id])
        if b.sum() == 1:
            return id
        else:
            return np.where(b == True)

    def is_p1_larger_than_p2(self, p1_id, p2_id):
        """
            Checks if an index of a point in Star is larger than an index of other point.
        
            p1_id : index of point 1
            p2_id : index of point 2
            
            return:
                True -> if there exists the case that p1 >= p2
                False -> if there is no case that p1 >= p2
        """
        if p1_id < 0 or p1_id >= self.dim:
            raise Exception('error: Invalid index for point 1')

        if p2_id < 0 or p2_id >= self.dim:
            raise Exception('error: Invalid index for point 2')

        d1 = self.V[p1_id, 0] - self.V[p2_id, 0]
        C1 = self.V[p2_id, 1:self.nVar+1] - self.V[p1_id, 1:self.nVar+1]

        new_C = np.vstack([self.C, C1])
        new_d = np.hstack([self.d, d1])
        S = Star(self.V, new_C, new_d, self.predicate_lb, self.predicate_ub)

        if S.isEmptySet():
            return False
        else:
            return True

    #------------------check if this function is working--------------------------------------------
    # find a oriented box bound a star
    def getOrientedBox(self):
        # !!! the sign of SVD result is different compared to Matalb
        # The sign of U and V is different from Matlab
        # bounds are correct from Gurobi
        # [Q, sdiag, vh] = np.linalg.svd(self.V[:, 1:])
        U, S, V = np.linalg.svd(self.V[:, 1:])
        print('U: ', U)
        print('S: ', S)
        print('V: ', V)
        Z = np.zeros([len(S), self.nVar])
        print('Z: ', Z)
        np.fill_diagonal(Z, S)
        print('Z: ', Z)
        
        S = Z @ V
        print('S: ', S)
        
        lb = np.zeros(self.dim)
        ub = np.zeros(self.dim)

        for i in range(self.dim):
            f = S[i, :]

            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            x = min_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(self.C)
            min_.addConstr(C @ x <= self.d)
            min_.optimize()

            if min_.status == 2:
                lb[i] = min_.objVal
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            x = max_.addMVar(shape=self.nVar, lb=self.predicate_lb, ub=self.predicate_ub)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(self.C)
            max_.addConstr(C @ x <= self.d)
            max_.optimize()

            if max_.status == 2:
                ub[i] = max_.objVal
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))

        print('lb: ', lb)
        print('ub: ', ub)
        print('self.V[:, 0]: ', self.V[:, 0])
        print('U: ', U)
        print('-U: ', -U)
        new_V = np.hstack([self.V[:, 0].reshape(-1, 1), -U])
        new_C = np.vstack([np.eye(self.dim), -np.eye(self.dim)])
        new_d = np.hstack([ub, -lb])
        
        print('new_V: ', new_V)
        print('new_C: ', new_C)
        print('new_d: ', new_d)
        return Star(new_V, new_C, new_d)

    def getZono(self):
        """
            Finds a zonotope bounding of a Star (an over-approximation of a Star using zonotope)
            
            return -> created Zono
        """   
        from box import Box
        B = self.getBox()
        if isinstance(B, Box):
            return B.toZono()
        else:
            return np.array([])

    def concatenate(self, X):
        """
            Concatenates the current Star with other Star, X
            X : input star
            
            return -> concataned Star
        """
        assert isinstance(X, Star), 'Input X is not a Star'
        
        c = np.hstack([self.V[:, 0], X.V[:, 0]]).reshape(-1, 1)
        V = scipy.linalg.block_diag(self.V[:, 1:], X.V[:, 1:])

        new_V = np.hstack([c, V])
        new_C = scipy.linalg.block_diag(self.C, X.C)
        new_d = np.hstack([self.d, X.d])

        if self.predicate_lb.size and X.predicate_lb.size:
            new_predicate_lb = np.hstack([self.predicate_lb, X.predicate_lb])
            new_predicate_ub = np.hstack([self.predicate_ub, X.predicate_ub])
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)
        return Star(new_V, new_C, new_d)

    def concatenate_with_vector(self, v):
        """
            Concatenates a Star with a vector
            v : a vector (1D numpy array)
            
            return -> concataned Star
        """
        assert len(v.shape) == 1, 'error: a vector should be 1D numpy array'
        
        c = np.hstack([v, self.V[:, 0]]).reshape(-1, 1)
        V = np.vstack([np.zeros([v.shape[0], self.nVar]), self.V[:, 1:]])
        new_V = np.hstack([c, V])
        return Star(new_V, self.C, self.d, self.predicate_lb, self.predicate_ub)

    def rand(dim, N):
        """
            Generates random star set
            dim : dimension of the random star set
            N : maximum number of constraints in polytope
            
            return -> generated star set
        """
        assert dim > 0, "error: Invalid dimension"
        
        A = np.random.rand(N, dim)
        # compute the convex hull
        P = pc.qhull(A)
        return Star(P)
        
    def plot(self, color=""):
        """
            Plots a Star using Polytope package
            color : color of Polytope 
                (if color is not provided, then polytope is plotted in random color)
        """
        assert self.dim <= 2 and self.dim > 0, 'error: only 2D star can be plotted'
        
        [l, u] = self.estimateRanges()
        P = self.toPolytope()
        if color: ax = P.plot(color=color)
        else:     ax = P.plot()
        ax.set_xlim(l[0], u[0]) # Optional: set axis max/min
        ax.set_ylim(l[1], u[1])
        return ax

    def __str__(self):
        from zono import Zono
        print('class: %s' % self.__class__)
        print('V: [shape: %s | type: %s]' % (self.V.shape, self.V.dtype))
        print('C: [shape: %s | type: %s]' % (self.C.shape, self.C.dtype))
        print('d: [shape: %s | type: %s]' % (self.d.shape, self.d.dtype))
        print('dim: %s' % self.dim)
        print('nVar: %s' % self.nVar)
        if self.predicate_lb.size:
            print('predicate_lb: [shape: %s | type: %s]' % (self.predicate_lb.shape, self.predicate_lb.dtype))
        else:
            print('predicate_lb: []')
        if self.predicate_ub.size:
            print('predicate_ub: [shape: %s | type: %s]' % (self.predicate_ub.shape, self.predicate_ub.dtype))
        else:
            print('predicate_ub: []')
        if self.state_lb.size:
            print('state_lb: [shape: %s | type: %s]' % (self.state_lb.shape, self.state_lb.dtype))
        else:
            print('state_lb: []')
        if self.state_ub.size:
            print('state_ub: [shape: %s | type: %s]' % (self.state_ub.shape, self.state_ub.dtype))
        else:
            print('state_ub: []')
        if isinstance(self.Z, Zono):
            print('Z: [1x1 Zono]')
        else:
            print('Z: []')
        return '\n'

    def __repr__(self):
        from zono import Zono
        if isinstance(self.Z, Zono):
            return "class: %s \nV: %s \nC: %s \nd: %s \ndim: %s \nnVar: %s \npredicate_lb: %s \npredicate_ub: %s \nstate_lb: %s \nstate_ub: %s\nZ: %s" % (self.__class__, self.V, self.C, self.d, self.dim, self.nVar, self.predicate_lb, self.predicate_ub, self.state_lb, self.state_ub, self.Z.__class__)
        else:
            return "class: %s \nV: %s \nC: %s \nd: %s \ndim: %s \nnVar: %s \npredicate_lb: %s \npredicate_ub: %s \nstate_lb: %s \nstate_ub: %s" % (self.__class__, self.V, self.C, self.d, self.dim, self.nVar, self.predicate_lb, self.predicate_ub, self.state_lb, self.state_ub)

    def get_hypercube_hull(stars):
        """
            Bounds a set of stars by a box
            stars : an 1D numpy array of stars
            
            return -> Box
        """
        from box import Box
        
        assert len(stars.shape) == 1, 'error: an input should be 1D numpy array that contains Stars '
        
        n = len(stars)
        dim = stars[0].dim
        
        for i in range(1, n):
            assert isinstance(stars[i], Star), 'error: The %d-th object is not a star' % i
            assert stars[i].dim == dim, 'error: Incosistent dimensions between stars'
        
        lb = np.empty([0,2])
        ub = np.empty([0,2])
        for i in range(n):
            B1 = stars[i].getBox()
            lb = np.vstack([lb, B1.lb])
            ub = np.vstack([ub, B1.ub])
        
        lb = np.min(lb, axis=0)
        ub = np.max(ub, axis=0)

        return Box(lb, ub)
    
    #------------------------------- Need to Test this function ---------------------------------#
    # def get_convex_hull():
    
    def concatenateStars(stars):
        """
            Concatenates many stars into a single star
            stars : an array of stars
            
            return -> a new Star
        """
        assert len(stars.shape) == 1, 'error: Input stars should be 1D numpy array of Stars'
        
        new_c = np.empty([0])
        new_V = np.empty([0, 0])
        new_C = np.empty([0, 0])
        new_d = np.empty([0])
        new_pred_lb = np.empty([0])
        new_pred_ub = np.empty([0])
        
        n = len(stars)
        for i in range(n):
            assert isinstance(stars[i], Star), 'error: The %d-th input is not a Star' % i
            new_c = np.hstack([new_c, stars[i].V[:, 0]])
            new_V = scipy.linalg.block_diag(new_V, stars[i].V[:, 1:])
            new_C = scipy.linalg.block_diag(new_C, stars[i].C)
            new_d = np.hstack([new_d, stars[i].d])
            new_pred_lb = np.hstack([new_pred_lb, stars[i].predicate_lb])
            new_pred_ub = np.hstack([new_pred_ub, stars[i].predicate_ub])
        
        return Star(np.hstack([new_c.reshape(-1, 1), new_V]), new_C, new_d, new_pred_lb, new_pred_ub)
    
    #------------------------------- Need to Test this function ---------------------------------#
    # def merge_stars()