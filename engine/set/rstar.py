#!/usr/bin/python3
import numpy as np
from numpy.core.numeric import Inf
import scipy
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB


class RStar:
    # Relaxed Star (RStar) set class
    # Note: abstract domain method is used to compute predicate bounds instead of linear programming solver
    # Star set defined by x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
    #                       = V * b,
    #                     V = [c v[1] v[2] ... v[n]]
    #                     b = [1 a[1] a[2] ... a[n]]^T
    #                     where C*a <= d, constraints on a[i]
    # D_L - lower polyhedral constraint
    # D_U - upper polyhedral constraint
    # lb - lower polyhedral bound
    # ub - upper polyhedral bound
    # author: Sung Woo Choi
    # date: 10/21/2021

    def __init__(obj,                   # RStar
                V = np.array([]),       # basic matrix
                C = np.array([]),       # predicate constraint matrix
                d = np.array([]),       # predicate constraint vector
                pred_lb = np.array([]), # predicate lower bound
                pred_ub = np.array([]), # predicate upper bound
                D_L = [],               # lower polyhedral domain constraints
                D_U = [],               # upper polyhedral domain constraints
                lb = [],                # lower polyhedral domain bounds
                ub = [],                # upper polyhedral domain bounds
                iter = Inf,             # number of iterations for back substitution
                ):
        from engine.set.star import Star

        assert isinstance(V, np.ndarray), 'error: basic matrix is not an ndarray'
        assert isinstance(C, np.ndarray), 'error: constraint matrix is not an ndarray'
        assert isinstance(d, np.ndarray), 'error: constraint vector is not an ndarray'
        assert isinstance(D_L, list), 'error: lower polyhedral domain constratins must be a list'
        assert isinstance(D_U, list), 'error: upper polyhedral domain constratins must be a list'
        
        if not V.dtype == C.dtype == d.dtype == np.float64:
            V = V.astype('float64')
            C = C.astype('float64')
            d = d.astype('float64')

        assert iter > 0, 'error: number of iteration for back-substitution must be greater than zero'
        obj.iter = iter

        if isinstance(lb, np.ndarray) and isinstance(ub, np.ndarray):
            assert lb.shape == ub.shape, 'error: inconsistency between lower and upper polyhedral domain bounds'
            if not lb.dtype == ub.dtype == np.float64:
                lb = lb.astype('float64')
                ub = ub.astype('float64')
            lb = [lb]
            ub = [ub]
        # elif not (isinstance(ub, list) and isinstance(lb, list)):
        elif not isinstance(ub, list) and not isinstance(lb, list):
            raise Exception('error: lower and upper bounds must be either ndarray or list')

        if V.size and C.size and d.size and pred_lb.size and pred_ub.size and D_U and D_L and lb and ub:
            assert V.shape[1] == C.shape[1] + 1, 'error: inconsistency between basic matrix and constraint matrix'
            assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and constraint vector'
            assert d.shape[1] == 1, 'error: constraint vector should have one column'
            assert len(D_L) == len(D_U) == len(lb) == len(ub), 'error: inconsistent number of D_L, D_U, lb, and ub' 
            assert pred_lb.shape == pred_ub.shape, 'error: inconsistency between predicate lower and uppper bounds'
            assert pred_lb.shape[0] == C.shape[1], 'error: inconcistency number of predicate variables'
            n = len(lb) - 1
            assert lb[n].shape == ub[n].shape, 'error: inconsistency between lower and upper bounds'
            assert D_L[n].shape == D_U[n].shape, 'error: inconsistency between D_L and D_U'
            obj.V = V
            obj.C = C
            obj.d = d
            obj.dim = V.shape[0]
            
            obj.predicate_lb = pred_lb
            obj.predicate_ub = pred_ub

            obj.D_L = D_L
            obj.D_U = D_U
            obj.lb = lb
            obj.ub = ub
            return

        if V.size and C.size and d.size and D_U and D_L and lb and ub:
            assert V.shape[1] == C.shape[1] + 1, 'error: inconsistency between basic matrix and constraint matrix'
            assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and constraint vector'
            assert d.shape[1] == 1, 'error: constraint vector should have one column'
            assert len(D_L) == len(D_U) == len(lb) == len(ub), 'error: inconsistent number of D_L, D_U, lb, and ub' 
            n = len(lb) - 1
            assert lb[n].shape == ub[n].shape, 'error: inconsistency between lb and ub'
            assert D_L[n].shape == D_U[n].shape, 'error: inconsistency between D_L and D_U'

            obj.V = V
            obj.C = C
            obj.d = d
            obj.dim = V.shape[0]

            obj.predicate_lb = -np.ones((obj.dim, 1))
            obj.predicate_ub = np.ones((obj.dim, 1))

            # obj.predicate_lb = obj.lb[0::2]
            # obj.predicate_ub = obj.ub[0::2]

            obj.D_L = D_L
            obj.D_U = D_U
            obj.lb = lb
            obj.ub = ub
            return
        
        if V.size and C.size and d.size:
            assert V.shape[1] == C.shape[1] + 1, 'error: inconsistency between basic matrix and constraint matrix'
            assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and constraint vector'
            assert d.shape[1] == 1, 'error: constraint vector should have one column'
            
            S = Star(V, C, d)
            [lb, ub] = S.getRanges()

            obj.V = V
            obj.C = C
            obj.d = d
            obj.dim = S.dim

            obj.predicate_lb = -np.ones((S.dim, 1))
            obj.predicate_ub = np.ones((S.dim, 1))

            obj.D_L = [np.column_stack((np.zeros((S.dim, 1)), np.eye(S.dim)))]
            obj.D_U = [np.column_stack((np.zeros((S.dim, 1)), np.eye(S.dim)))]
            obj.lb = [lb]
            obj.ub = [ub]
            
            return

        if lb and ub and D_L and D_U:
            assert len(D_L) == len(D_U) == len(lb) == len(ub), 'error: inconsistent number of D_L, D_U, lb, and ub' 
            n = len(lb) - 1
            assert lb[n].shape == ub[n].shape, 'error: inconsistency between lb and ub'
            assert D_L[n].shape == D_U[n].shape, 'error: inconsistency between D_L and D_U'
            
            S = Star(lb = lb[n], ub = ub[n])
            obj.V = S.V
            obj.C = S.C
            obj.d = S.d
            obj.dim = S.dim

            obj.D_L = D_L
            obj.D_U = D_U
            obj.lb = lb
            obj.ub = ub
            return

        if lb and ub:
            n = len(lb) - 1
            assert lb[n].shape == ub[n].shape, 'error: inconsistency between lower and upper bounds'
            assert len(lb) == len(ub) == 1, 'error: lb and ub should be lists containing one array'
            
            S = Star(lb = lb[n], ub = ub[n])
            obj.V = S.V
            obj.C = S.C
            obj.d = S.d
            obj.dim = S.dim

            obj.predicate_lb = -np.ones((S.dim, 1))
            obj.predicate_ub = np.ones((S.dim, 1))

            obj.D_L = [np.column_stack((np.zeros((obj.dim, 1)), np.eye(obj.dim)))]
            obj.D_U = [np.column_stack((np.zeros((obj.dim, 1)), np.eye(obj.dim)))]
            obj.lb = lb
            obj.ub = ub
            return

    # check if RStar is an empty set
    def isEmptySet(obj):
        # error code (m.status) description avaliable at
        # https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
        # parameter settings: https://www.gurobi.com/documentation/9.1/refman/parameters.html

        f = np.zeros((1, obj.C.shape[1]))
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        x = m.addMVar(shape=obj.C.shape[1], lb=obj.predicate_lb, ub=obj.predicate_ub)
        m.setObjective(f @ x, GRB.MINIMIZE)
        A = sp.csr_matrix(obj.C)
        b = np.array(obj.d).flatten()
        m.addConstr(A @ x <= b)     
        m.optimize()

        if m.status == 2:
            return False
        elif m.status == 3:
            return True
        else:
            raise Exception('error: exitflat = %d' % (m.status))
    
    # check if a star set contain a point
    def contains(obj, s):
        # @s: a star point (column vector)
        # return: = 1 star set contains s; = 0 star set does not contain s; else error code
        assert s.shape[0] == obj.dim, 'error: dimension mismatch'
        assert s.shape[1] == 1, 'error: invalid star point'
        f = np.zeros((1, obj.C.shape[1]))
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        x = m.addMVar(shape=obj.C.shape[1], lb=obj.predicate_lb, ub=obj.predicate_ub)
        m.setObjective(f @ x, GRB.MINIMIZE)
        A = sp.csr_matrix(obj.C)
        b = np.array(obj.d).flatten()
        m.addConstr(A @ x <= b)
        Ae = sp.csr_matrix(obj.V[:, 1:])
        be = s.flatten() - obj.V[:, 0]
        m.addConstr(Ae @ x == be)
        m.optimize()

        if m.status == 2:
            return True
        elif m.status == 3:
            return False
        else:
            raise Exception('error: exitflat = %d' % (m.status))

    # sampling a star set
    def sample(obj, N):
        # @N: number of points in the sample
        # return: V: a set of at most N sampled points in the star set
        from engine.set.box import Box
        assert N >= 1, 'error: invalid number of samples'

        [lb, ub] = obj.getRanges()
        V1 = np.array([])
        for i in range(obj.dim):
            X = (ub[i] - lb[i]) * np.random.rand(2*N, 1) + lb[i]
            V1 = np.hstack((V1, X)) if V1.size else X
            
        V = np.array([])
        for i in range(2*N):
            v1 = V1[i,:].reshape(-1,1)
            if obj.contains(v1):
                V = np.hstack((V, v1)) if V.size else v1

        if V.shape[1] >= N:
            V = V[:, 0:N]
        return V

    # Minkowski Sum
    def Sum(obj, X):
        # @X: another RStar with the same dimension
        # return: new RStar = (obj (+) X), where (+) is Minkowski Sum
        assert isinstance(X, RStar), 'error: input set X is not a RStar'
        assert X.dim == obj.dim, 'error: inconsistent dimension between object rstar and input rstar X'
        assert len(X.lb) == len(obj.lb) and len(X.ub) == len(obj.ub) and len(X.lb) == len(X.ub), 'error: inconsistent length of lb and ub between object rstar and input rstar X' 
        assert len(X.D_L) == len(obj.D_L) and len(X.D_U) == len(obj.D_U) and len(X.D_L) == len(X.D_U), 'error: inconsistent length of D_L and D_U between object rstar and input rstar X'

        V1 = np.hstack((obj.V[:, 1:], X.V[:, 1:]))
        new_c = (obj.V[:, 0] + X.V[:, 0]).reshape(-1, 1)
        new_V = np.hstack((new_c, V1))
        new_C = scipy.linalg.block_diag(obj.C, X.C)
        new_d = np.vstack((obj.d, X.d))
        new_pred_lb = np.vstack((obj.predicate_lb, X.predicate_lb))
        new_pred_ub = np.vstack((obj.predicate_ub, X.predicate_ub))

        new_D_L, new_D_U, new_lb, new_ub = [],[],[],[]
        n = len(obj.D_L)
        for i in range(n):
            temp_D_L = scipy.linalg.block_diag(obj.D_L[i], X.D_L[i][:, 1:])
            temp_D_L[obj.D_L[i].shape[0]:, 0] = X.D_L[i][:, 0].reshape(-1)
            new_D_L.append(temp_D_L)

            temp_D_U = scipy.linalg.block_diag(obj.D_U[i], X.D_U[i][:, 1:])
            temp_D_U[obj.D_U[i].shape[0]:, 0] = X.D_U[i][:, 0].reshape(-1)
            new_D_U.append(temp_D_U)
            
            new_lb.append(np.vstack((obj.lb[i], X.lb[i])))
            new_ub.append(np.vstack((obj.ub[i], X.ub[i])))

        return RStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_D_L, new_D_U, new_lb, new_ub, obj.iter)

    # affine mapping of RStar set R = Wx + b
    def affineMap(obj, W, b = np.array([])):
        # @W: mapping matrix
        # @b: mapping vector
        # return a new RStar set
        assert isinstance(W, np.ndarray), 'error: weight matrix is not an matrix'
        assert isinstance(b, np.ndarray), 'error: bias vector is not an matrix'

        [nb, mb] = b.shape
        [nW, mW] = W.shape

        assert mW == obj.dim, 'error: inconsistent dimension between weight matrix with the zonotope dimension'
        assert mb <= 1, 'error: bias vector must be one column'
        assert nW == nW and nb > 0, 'error: inconsistency between the affine mapping matrix and the bias vector'

        # affine mapping of basic matrix
        V = W @ obj.V
        if mb != 0:
            V[:, 0] = V[:, 0] + b
        else:
            # if b is not provided, then create zero bias vector which dimesion corresponds to dimesion of weight matrix
            b = np.zeros((nW, 1))
        
        # new lower and upper polyhedral contraints
        D_L, D_U = obj.D_L, obj.D_U
        lb, ub = obj.lb, obj.ub

        D_L.append(np.hstack((b, W)))
        D_U.append(np.hstack((b, W)))
        lb.append(obj.lb_backSub(D_L, D_U))
        ub.append(obj.ub_backSub(D_L, D_U))

        return RStar(V, obj.C, obj.d, obj.predicate_lb, obj.predicate_ub, D_L, D_U, lb, ub, obj.iter)

    # lower bound back-substitution
    def lb_backSub(obj, D_L, D_U):
        n = len(obj.D_L) - 1
        nL = D_L[n].shape[0]
        A = D_L[n][:, 1:]
        bl = D_L[n][:, 0]
        bu = np.zeros((nL, 1))
        # print("lb_backSub-----------------------------------------\n")
        # print("DL[n]: %s" % (D_L[n]))
        # print("A: %s" % (A))
        # print("bl: %s" % (bl))
        # print("bu: %s" % (bu))
        # print("n: ", n)
        # print("-----------------------------------------\n")
        n = n - 1
        iter = 0
        while (n > 0 & iter < obj.iter):
            nL = D_L[n].shape[0]
            max_A = np.where(A > 0, A, 0)
            min_A = np.where(A < 0, A, 0)
            
            bl = max_A @ D_L[n][:, 1] + bl
            bu = min_A @ D_U[n][:, 1] + bu

            A = max_A @ D_L[n][:, 1:] + min_A @ D_U[n][:, 1:]
            n = n - 1
            iter = iter + 1

            # print("A: %s" % (A))
            # print("bl: %s" % (bl))
            # print("bu: %s" % (bu))
            # print("-----------------------------------------\n")
        
        max_A = np.where(A > 0, A, 0)
        min_A = np.where(A < 0, A, 0)

        [lb1, ub1] = obj.getRanges_L(n)
        # print("max_A: ", max_A)
        # print("min_A: ", min_A)
        # print("lb1: ", lb1)
        # print("ub1: ", ub1)
        # print("np.matmul(max_A, lb1) + bl: ", np.matmul(max_A, lb1) + bl)
        # print("l: ", np.matmul(max_A, lb1) + bl + np.matmul(min_A, ub1) + bu)
        return max_A @ lb1 + bl + min_A @ ub1 + bu

    # upper bound back-substitution
    def ub_backSub(obj, D_L, D_U):
        n = len(obj.D_U) - 1
        nL = D_U[n].shape[0]
        A = D_U[n][:, 1:]
        bl = np.zeros((nL, 1))
        bu = D_U[n][:, 0]

        n = n - 1
        iter = 0
        while (n > 0 & iter < obj.iter):
            [nL, mL] = D_U[n].shape
            max_A = np.where(A > 0, A, 0)
            min_A = np.where(A < 0, A, 0)
            
            bl = min_A @ D_L[n][:, 1] + bl
            bu = max_A @ D_U[n][:, 1] + bu

            A = min_A @ D_L[n][:, 1:] + max_A @ D_U[n][:, 1:]
            n = n - 1
            iter = iter + 1
        
        max_A = np.where(A > 0, A, 0)
        min_A = np.where(A < 0, A, 0)

        [lb1, ub1] = obj.getRanges_L(n)
        return min_A @ lb1 + bl + max_A @ ub1 + bu

    # get the lower and upper bounds of a current layer at specific position
    def getRange(obj, i):
        assert i >= 0 and i <= obj.dim, 'error: invalid index'
        n = len(obj.lb) - 1
        lb = obj.lb[n][i]
        ub = obj.ub[n][i]
        return [lb, ub]

    # get lower and upper bounds of a current layer
    def getRanges(obj):
        n = len(obj.lb) - 1
        lb = obj.lb[n]
        ub = obj.ub[n]

        m = lb.shape[0]
        dim = obj.dim
        l = np.zeros((dim, 1))
        u = np.zeros((dim, 1))
        if dim != m:
            for i in range(m//dim):
                j = i*dim
                l += lb[j : j+dim]
                u += ub[j : j+dim]
            return [l, u]
        return[lb, ub]
    
    # get lower and upper bounds of a specific layer
    def getRanges_L(obj, numL):
        n = len(obj.D_L) - 1
        assert numL <= n, 'error: range request should be layer with interation'
        lb = obj.lb[numL]
        ub = obj.ub[numL]
        return [lb, ub]

    # get exact lower and uppper bounds
    def getExactRanges(obj):
        if not obj.isEmptySet():
            n = obj.dim
            lb = np.zeros((n, 1))
            ub = np.zeros((n, 1))
            for i in range(n):
                [lb[i], ub[i]] = obj.getExactRange(i)
        else:
            lb = np.array([])
            ub = np.array([])
        return [lb, ub]

    # get exact lower and uppper bounds at specific position using LP solver
    def getExactRange(obj, index):
        # @index: position of the state
        # return: ranges; min and max values of x[index]

        assert index >= 0 and index < obj.dim, 'error: invalid index'

        f = obj.V[index, 1:]
        if (f == 0).all():
            xmin = obj.V[index, 0]
            xmax = obj.V[index, 0]
        else:
            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            x = min_.addMVar(shape=obj.C.shape[1], lb=obj.predicate_lb, ub=obj.predicate_ub)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(obj.C)
            d = np.array(obj.d).flatten()
            min_.addConstr(C @ x <= d)
            min_.optimize()

            if min_.status == 2:
                xmin = min_.objVal + obj.V[index, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            x = max_.addMVar(shape=obj.C.shape[1], lb=obj.predicate_lb, ub=obj.predicate_ub)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(obj.C)
            d = np.array(obj.d).flatten()
            max_.addConstr(C @ x <= d)
            max_.optimize()

            if max_.status == 2:
                xmax = max_.objVal + obj.V[index, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))

        return [xmin, xmax]

    # convert to Polyhedron
    # def toPolyhedron(obj):
        
    # convert to Zonotope
    def toZono(obj):
        from engine.set.zono import Zono
        return Zono(obj.c, obj.X)

    # convert to Star
    def toStar(obj):
        from engine.set.star import Star
        return Star(obj.V, obj.C, obj.d, obj.pred_lb, obj.pred_ub)

    # get basic matrix
    # @property
    # def X(obj):
    #     return obj.V[:, 1:]

    # get center vector
    # @property
    # def c(obj):
    #     return obj.V[:, 1]

    # # get predicate lower bound
    # @property
    # def predicate_lb(obj):
    #     return obj.lb[0::2]

    # # get predicate upper bound
    # @property
    # def predicate_ub(obj):
    #     return obj.ub[0::2]

    # check if a index is larger than other
    def is_p1_larger_than_p2(obj, p1_id, p2_id):
        # @p1_id: index of point 1
        # @p2_id: index of point 2
        # return: bool = 1 if there exists the case that p1 >= p2
        #                2 if there is no case that p1 >= p2
        from engine.set.star import Star

        assert p1_id >= 1 and p1_id <= obj.dim, 'error: invalid index for point 1'
        assert p2_id >= 1 and p2_id <= obj.dim, 'error: invalid index for point 2'

        d1 = obj.c[p1_id] - obj.c[p2_id]
        C1 = obj.X[p2_id, :] - obj.X[p1_id, :]
        S = Star(obj.V, np.vstack((obj.C, C1)), obj.vstack((obj.d, d1)), obj.pred_lb, obj.pred_ub)
        if S.isEmptySet:
            return 0
        else:
            return 1

    def __str__(obj):
        print('class: %s' % obj.__class__)
        print('V: [%sx%s %s]' % (obj.V.shape[0], obj.V.shape[1], obj.V.dtype))
        print('C: [%sx%s %s]' % (obj.C.shape[0], obj.C.shape[1], obj.C.dtype))
        print('d: [%sx%s %s]' % (obj.d.shape[0], obj.d.shape[1], obj.d.dtype))
        print('predicate_lb: [%sx%s %s]' % (obj.predicate_lb.shape[0], obj.predicate_lb.shape[1], obj.predicate_lb.dtype))
        print('predicate_ub: [%sx%s %s]' % (obj.predicate_ub.shape[0], obj.predicate_ub.shape[1], obj.predicate_ub.dtype))
        print('D_L: [', end='')
        for matrix in obj.D_L:
            print('[%sx%s]' % (matrix.shape[0], matrix.shape[1]), end='')
        print(']')
        print('D_U: [', end='')
        for matrix in obj.D_U:
            print('[%sx%s]' % (matrix.shape[0], matrix.shape[1]), end='')
        print(']')
        print('lb: [', end='')
        for matrix in obj.lb:
            print('[%sx%s]' % (matrix.shape[0], matrix.shape[1]), end='')
        print(']')
        print('ub: [', end='')
        for matrix in obj.ub:
            print('[%sx%s]' % (matrix.shape[0], matrix.shape[1]), end='')
        print(']')
        print('iter: %s' % obj.iter)
        print('dim: %s' % obj.dim)
        return ''

    def __repr__(obj):
        return "class: %s \nV: %s \nC: %s \nd: %s \npredicate_lb: %s \npredicate_ub: %s\nD_L: %s \nD_U: %s \nlb: %s \nub: %s \niter: %s \ndim: %s" % (obj.__class__, obj.V, obj.C, obj.d, obj.predicate_lb, obj.predicate_ub, obj.D_L, obj.D_U, obj.lb, obj.ub, obj.iter, obj.dim)