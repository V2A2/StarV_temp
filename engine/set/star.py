#!/usr/bin/python3
import numpy as np
import scipy
import scipy.sparse as sp
import gurobipy as gp
from gurobipy import GRB
import polytope as pc
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as colors

class Star:
    # Star set class
    # Star set defined by x = c + a[1]*v[1] + a[2]*v[2] + ... + a[n]*v[n]
    #                       = V * b,
    #                     V = [c v[1] v[2] ... v[n]]
    #                     b = [1 a[1] a[2] ... a[n]]^T
    #                     where C*a <= d, constraints on a[i]
    # author: Sung Woo Choi
    # date: 9/21/2021

    def __init__(obj,
                V = np.array([]),          # basic matrix
                C = np.array([]),          # constraint matrix
                d = np.array([]),          # constraint vector
                pred_lb = np.array([]),    # lower bound vector of predicate variable
                pred_ub = np.array([]),    # upper bound vector of predicate variable
                state_lb = np.array([]),   # lower bound of state variables
                state_ub = np.array([]),   # upper bound of state variables
                outer_zono = np.array([]), # an outer zonotope covering this star, used for reachability of logsig and tansig networks
                lb = np.array([]),
                ub = np.array([])):
        from engine.set.zono import Zono
        from engine.set.box import Box

        assert isinstance(V, np.ndarray), 'error: basic matrix is not an ndarray'
        assert isinstance(C, np.ndarray), 'error: constraint matrix is not an ndarray'
        assert isinstance(d, np.ndarray), 'error: constraint vector is not an ndarray'
        assert isinstance(pred_lb, np.ndarray), ' error: predicate lower bound is not an ndarray'
        assert isinstance(pred_ub, np.ndarray), 'error: predicate upper bound is not an ndarray'
        assert isinstance(state_lb, np.ndarray), 'error: state lower bound is not an ndarray'
        assert isinstance(state_ub, np.ndarray), 'error: state upper bound is not an ndarray'
        assert isinstance(lb, np.ndarray), 'error: lower bound vector is not an ndarray'
        assert isinstance(ub, np.ndarray), 'error: upper bound vector is not an ndarray'

        V = V.astype('float64')
        C = C.astype('float64')
        d = d.astype('float64')
        lb = lb.astype('float64')
        ub = ub.astype('float64')

        if V.size and C.size and d.size:
            assert V.shape[1] == C.shape[1] + 1, 'error: inconsistency between basic matrix and constraint matrix'
            assert C.shape[0] == d.shape[0], 'error: inconsistency between constraint matrix and constraint vector'
            assert d.shape[1] == 1, 'error: constraint vector should have one column'

            if isinstance(outer_zono, Zono):
                assert outer_zono.V.shape[0] == V.shape[0], 'error: inconsistent dimension between outer zonotope and star set'
                obj.Z = outer_zono
            else:
                obj.Z = np.array([])

            obj.V = V
            obj.C = C
            obj.d = d
            obj.dim = V.shape[0]    # dimension of star set
            obj.nVar = C.shape[1]   # number of variable in the constraints

            if state_lb.size and state_ub.size:
                assert state_lb.shape[0] == state_ub.shape[0] == V.shape[0], 'error: inconsistent dimension between lower bound and upper bound vector of state variables and matrix V'
                assert state_lb.shape[1] == state_ub.shape[1] == 1, 'error: invalid lower bound or upper bound vector of state variables'
                obj.state_lb = state_lb
                obj.state_ub = state_ub
            else:
                obj.state_lb = np.matrix([])
                obj.state_ub = np.matrix([])

            if pred_lb.size and pred_ub.size:
                assert pred_lb.shape[1] == pred_ub.shape[1] == 1, 'error: predicate lower- or upper-bounds vector should have one column'
                assert pred_lb.shape[0] == pred_ub.shape[0] == C.shape[1], 'error: inconsistency between number of predicate variables and predicate lower- or upper-bounds vector'
                obj.predicate_lb = pred_lb
                obj.predicate_ub = pred_ub
            else:
                obj.predicate_lb, obj.predicate_ub = -np.ones((obj.nVar, 1)), np.ones((obj.nVar, 1))
            return

        if lb.size and ub.size:
            assert lb.shape[0] == ub.shape[0], 'error: inconsistency between upper and lower bounds'
            
            B = Box(lb, ub)
            S = B.toStar()
            obj.V = S.V
            obj.C = np.zeros((1, S.nVar)) # initiate an obvious constraint
            obj.d = np.zeros((1, 1))
            obj.dim = S.dim
            obj.nVar = S.nVar
            obj.state_lb = lb
            obj.state_ub = ub
            obj.predicate_lb = -np.ones((S.nVar, 1))
            obj.predicate_ub = np.ones((S.nVar, 1))
            obj.Z = B.toZono()
            return

        raise Exception('error: failed to create Star set')

    # check is empty set
    def isEmptySet(obj):
        # error code (m.status) description avaliable at
        # https://www.gurobi.com/documentation/9.1/refman/optimization_status_codes.html
        # parameter settings: https://www.gurobi.com/documentation/9.1/refman/parameters.html

        f = np.zeros((1, obj.nVar))
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if obj.predicate_lb.size and obj.predicate_ub.size:
            x = m.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
        else:
            x = m.addMVar(shape=obj.nVar)
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
        f = np.zeros((1, obj.nVar))
        m = gp.Model()
        # prevent optimization information
        m.Params.LogToConsole = 0
        m.Params.OptimalityTol = 1e-9
        if obj.predicate_lb.size and obj.predicate_ub.size:
            x = m.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
        else:
            x = m.addMVar(shape=obj.nVar)
        m.setObjective(f @ x, GRB.MINIMIZE)
        A = sp.csr_matrix(obj.C)
        b = np.array(obj.d).flatten()
        m.addConstr(A @ x <= b)
        Ae = sp.csr_matrix(obj.V[:, 1:obj.nVar + 1])
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

        B = obj.getBox()
        if not isinstance(B, Box):
            V = np.array([])
        else:
            lb = B.lb
            ub = B.ub
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

    # affine mapping of star set S = Wx + b
    def affineMap(obj, W, b = np.array([])):
        # @W: mapping matrix
        # @b: mapping vector
        # return a new star set

        assert isinstance(W, np.ndarray), 'error: weight matrix is not an matrix'
        assert isinstance(b, np.ndarray), 'error: bias vector is not an matrix'
        assert W.shape[1] == obj.dim, 'error: inconsistent dimension between weight matrix with the zonotope dimension'

        W = np.matrix(W.astype('float64'))
        b = np.matrix(b.astype('float64'))

        if b.size:
            assert b.shape[1] == 1, 'error: bias vector should be a column vector'
            assert W.shape[0] == b.shape[0], 'error: inconsistency between weight matrix and bias vector'

            new_V = W @ obj.V
            new_V[:, 0] += b
        else:
            new_V = W @ obj.V

        if obj.Z:
            new_Z = obj.Z.affineMap(W, b)
        else:
            new_Z = np.array([])

        return Star(new_V, obj.C, obj.d, obj.predicate_lb, obj.predicate_ub, outer_zono = new_Z)

    # New Minkowski Sum
    def Sum(obj, X):
        # @X: another star with the same dimension
        # return: new star = (obj (+) X), where (+) is Minkowski Sum

        assert isinstance(X, Star), 'error: input set X is not a Star'
        assert X.dim == obj.dim, 'error: inconsistent dimension between object star and input star X'

        V1 = obj.V[:, 1:]
        V2 = X.V[:, 1:]

        V3 = np.hstack((V1, V2))
        new_c = (obj.V[:, 0] + X.V[:, 0]).reshape(-1, 1)
        new_V = np.hstack((new_c, V3))
        new_C = scipy.linalg.block_diag(obj.C, X.C)
        new_d = np.vstack((obj.d, X.d))

        if obj.predicate_lb.size and X.predicate_lb.size:
            new_predicate_lb = np.vstack((obj.predicate_lb, X.predicate_lb))
            new_predicate_ub = np.vstack((obj.predicate_ub, X.predicate_ub))
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)
        return Star(new_V, new_C, new_d)

#------------------check if this function is working--------------------------------------------
    # intersection with a half space: H(x) := Hx <= g
    def intersectHalfSpace(obj, H, g):
        # @H: HalfSpace matrix
        # @g: HalfSpace vector
        # return a new star set with more constraints

        assert isinstance(H, np.matrix), 'error: halfspace matrix is not an matrix'
        assert isinstance(g, np.matrix), 'error: halfspace vector is not an matrix'
        assert g.shape[1] == 1, 'error: halfspace vector should have one column'
        assert H.shape[0] == g.shape[0], 'inconsistent dimension between halfspace matrix and halfspace vector'
        assert H.shape[1] == obj.dim, 'inconsistent dimension between halfspace and star set'

        H = H.astype('float64')
        g = g.astype('float64')

        m = obj.V.shape[1]
        C1 = H @ obj.V[:, 1:m]
        d1 = g - H @ obj.V[:, 0]

        new_C = np.vstack((obj.C, C1))
        new_d = np.vstack((obj.d, d1))

        S = Star(obj.V, new_C, new_d, obj.predicate_lb, obj.predicate_ub)

        if S.isEmptySet:
            S = np.matrix([])
        return S

    # convert to ImageStar set
    def toImageStar(obj, height, width, numChannel):
        # @height: height of ImageStar
        # @width: width of ImageStar
        # @numChannel: number of channels in ImageStar
        # return: ImageStar
        from engine.set.imagestar import ImageStar

        assert obj.dim == height*width*numChannel, 'error: inconsistent dimension in the ImageStar and the original Star set'

        new_V = np.array(obj.V).reshape(obj.nVar + 1, numChannel, height, width)
        return ImageStar(new_V, obj.C, obj.d, obj.predicate_lb, obj.predicate_ub)

    # find a box bounding a star
    def getBox(obj):
        from engine.set.box import Box

        if obj.C.size == 0 or obj.d.size == 0 :
            # star set is just a vector (one point)
            lb = obj.V[:, 0]
            ub = obj.V[:, 0]
            return Box(lb, ub)
        else:
            # star set is a set
            if obj.state_lb.size and obj.state_ub.size:
                return Box(obj.state_lb, obj.state_ub)
            else:
                lb = np.zeros((obj.dim, 1))
                ub = np.zeros((obj.dim, 1))

                for i in range(obj.dim):
                    f = obj.V[i, 1:obj.nVar + 1]
                    if (f == 0).all():
                        lb[i] = obj.V[i, 0]
                        ub[i] = obj.V[i, 0]
                    else:
                        lb_ = gp.Model()
                        lb_.Params.LogToConsole = 0
                        lb_.Params.OptimalityTol = 1e-9
                        if obj.predicate_lb.size and obj.predicate_ub.size:
                            x = lb_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                        else:
                            x = lb_.addMVar(shape=obj.nVar)
                        lb_.setObjective(f @ x, GRB.MINIMIZE)
                        C = sp.csr_matrix(obj.C)
                        d = np.array(obj.d).flatten()
                        lb_.addConstr(C @ x <= d)
                        lb_.optimize()

                        if lb_.status == 2:
                            lb[i] = lb_.objVal + obj.V[i, 0]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (lb_.status))

                        ub_ = gp.Model()
                        ub_.Params.LogToConsole = 0
                        ub_.Params.OptimalityTol = 1e-9
                        if obj.predicate_lb.size and obj.predicate_ub.size:
                            x = ub_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                        else:
                            x = ub_.addMVar(shape=obj.nVar)
                        ub_.setObjective(f @ x, GRB.MAXIMIZE)
                        C = sp.csr_matrix(obj.C)
                        d = np.array(obj.d).flatten()
                        ub_.addConstr(C @ x <= d)
                        ub_.optimize()

                        if ub_.status == 2:
                            ub[i] = ub_.objVal + obj.V[i, 0]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (ub_.status))

                if lb.size == 0 or ub.size == 0:
                    return np.array([])
                else:
                    return Box(lb, ub)

    #find range of a state at specific position
    def getRange(obj, index):
        # @index: position of the state
        # range: min and max values of x[index]
        assert isinstance(index, int), 'error: index is not an integer'
        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid index')

        f = obj.V[index, 1:obj.nVar + 1]
        if (f == 0).all():
            xmin = obj.V[index, 0]
            xmax = obj.V[index, 0]
        else:
            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            if obj.predicate_lb.size and obj.predicate_ub.size:
                x = min_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
            else:
                x = min_.addMVar(shape=obj.nVar)
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
            if obj.predicate_lb.size and obj.predicate_ub.size:
                x = max_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
            else:
                x = max_.addMVar(shape=obj.nVar)
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

    # get min
    def getMin(obj, index, lp_solver = 'gurobi'):
        # @index: position of the state
        # xmin: min value of x[index]

        # assert isinstance(index, int), 'error: index is not an integer'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid index')

        f = obj.V[index, 1:obj.nVar + 1]
        if (f == 0).all():
            xmin = obj.V[index, 0]
        else:
            if lp_solver == 'gurobi':
                min_ = gp.Model()
                min_.Params.LogToConsole = 0
                min_.Params.OptimalityTol = 1e-9
                if obj.predicate_lb.size and obj.predicate_ub.size:
                    x = min_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                else:
                    x = min_.addMVar(shape=obj.nVar)
                min_.setObjective(f @ x, GRB.MINIMIZE)
                C = sp.csr_matrix(obj.C)
                d = np.array(obj.d).flatten()
                min_.addConstr(C @ x <= d)
                min_.optimize()

                if min_.status == 2:
                    xmin = min_.objVal + obj.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))
            else:
                raise Exception('error: unknown lp solver, should be gurobi')
        return xmin

#------------------check if this function is working--------------------------------------------
    # get mins
    def getMins(obj, map, par_option = 'single', dis_option = '', lp_solver = 'gurobi'):
        # @map: an array of indexes
        # xmin: min values of x[indexes]

        assert isinstance(map, np.ndarray), 'error: map is not a numpy ndarray'
        assert isinstance(par_option, str), 'error: par_option is not a string'
        assert isinstance(dis_option, str), 'error: dis_option is not a string'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        n = len(map)
        xmin = np.zeros((n, 1))
        if not len(par_option) or par_option == 'single':   # get Mins using single core
            reverseStr = ''
            for i in range(n):
                xmin[i] = obj.getMin(map[i], lp_solver)
# implement display option----------------------------------------------------------------------------------------
                # if dis_option = 'display':

                # msg = "%d/%d" % (i, n)
                # print([reverseStr, msg])
                # msg = sprintf('%d/%d', i, n);
                # fprintf([reverseStr, msg]);
                # reverseStr = repmat(sprintf('\b'), 1, length(msg));
# implement multiprogramming for parallel computation of bounds----------------------------------------------------
        elif par_option == 'parallel':  # get Mins using multiple cores
            f = obj.V[map, 1:obj.nVar + 1]
            V1 = obj.V[map, 0]
            for i in range (n):
                if f[i, :].sum() == 0:
                    xmin[i] = V1[i, 0]
                else:
                    if lp_solver == 'gurobi':
                        min_ = gp.Model()
                        min_.Params.LogToConsole = 0
                        min_.Params.OptimalityTol = 1e-9
                        if obj.predicate_lb.size and obj.predicate_ub.size:
                            x = min_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                        else:
                            x = min_.addMVar(shape=obj.nVar)
                        min_.setObjective(f[i, :] @ x, GRB.MINIMIZE)
                        C = sp.csr_matrix(obj.C)
                        d = np.array(obj.d).flatten()
                        min_.addConstr(C @ x <= d)
                        min_.optimize()

                        if min_.status == 2:
                            xmin[i] = min_.objVal + V1[i, 0]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))
        else:
            raise Exception('error: unknown lp solver, should be gurobi')
        return xmin

    # get max
    def getMax(obj, index, lp_solver = 'gurobi'):
        # @index: position of the state
        # xmax: max value of x[index]

        # assert isinstance(index, int), 'error: index is not an integer'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid index')

        f = obj.V[index, 1:obj.nVar + 1]
        if (f == 0).all():
            xmax = obj.V[index, 0]
        else:
            if lp_solver == 'gurobi':
                max_ = gp.Model()
                max_.Params.LogToConsole = 0
                max_.Params.OptimalityTol = 1e-9
                if obj.predicate_lb.size and obj.predicate_ub.size:
                    x = max_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                else:
                    x = max_.addMVar(shape=obj.nVar)
                max_.setObjective(f @ x, GRB.MAXIMIZE)
                C = sp.csr_matrix(obj.C)
                d = np.array(obj.d).flatten()
                max_.addConstr(C @ x <= d)
                max_.optimize()
                if max_.status == 2:
                    xmax = max_.objVal + obj.V[index, 0]
                else:
                    raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
            else:
                raise Exception('error: unknown lp solver, should be gurobi')
        return xmax

#------------------check if this function is working--------------------------------------------
    # get maxs
    def getMaxs(obj, map, par_option = 'single', dis_option = '', lp_solver = 'gurobi'):
        # @map: an array of indexes
        # xmax: max values of x[indexes]

        assert isinstance(map, np.ndarray), 'error: map is not a ndarray'
        assert isinstance(par_option, str), 'error: par_option is not a string'
        assert isinstance(dis_option, str), 'error: dis_option is not a string'
        assert isinstance(lp_solver, str), 'error: lp_solver is not a string'

        n = len(map)
        xmax = np.zeros((n, 1))
        if not len(par_option) or par_option == 'single':   # get Mins using single core
            reverseStr = ''
            for i in range(n):
                xmax[i] = obj.getMax(map[i], lp_solver)
# implement display option----------------------------------------------------------------------------------------
                # if dis_option = 'display':

                # msg = "%d/%d" % (i, n)
                # print([reverseStr, msg])
                # msg = sprintf('%d/%d', i, n);
                # fprintf([reverseStr, msg]);
                # reverseStr = repmat(sprintf('\b'), 1, length(msg));
# implement multiprogramming for parallel computation of bounds----------------------------------------------------
        elif par_option == 'parallel':  # get Mins using multiple cores
            f = obj.V[map, 1:obj.nVar + 1]
            V1 = obj.V[map, 0]
            for i in range (n):
                if f[i, :].sum() == 0:
                    xmax[i] = V1[i, 0]
                else:
                    if lp_solver == 'gurobi':
                        max_ = gp.Model()
                        max_.Params.LogToConsole = 0
                        max_.Params.OptimalityTol = 1e-9
                        if obj.predicate_lb.size and obj.predicate_ub.size:
                            x = max_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
                        else:
                            x = max_.addMVar(shape=obj.nVar)
                        max_.setObjective(f[i, :] @ x, GRB.MAXIMIZE)
                        C = sp.csr_matrix(obj.C)
                        d = np.array(obj.d).flatten()
                        max_.addConstr(C @ x <= d)
                        max_.optimize()
                        if max_.status == 2:
                            xmax[i] = max_.objVal + V1[i, 0]
                        else:
                            raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
        else:
            raise Exception('error: unknown lp solver, should be gurobi')
        return xmax

    # get lower bound and upper bound vector of the state variables
    def getRanges(obj):
        if not obj.isEmptySet():
            n = obj.dim
            lb = np.zeros((n, 1))
            ub = np.zeros((n, 1))
            for i in range(n):
                [lb[i], ub[i]] = obj.getRange(i)
        else:
            lb = np.matrix([])
            ub = np.matrix([])
        return [lb, ub]

    # find range of a state at specific position
    def estimateRange(obj, index):
        # @index: estimateRange(obj, index)
        # range: min and max values of x[index]

        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid index')
            
        if obj.predicate_lb.size and obj.predicate_ub.size:
            f = obj.V[index, :]
            xmin = f.item(0)
            xmax = f.item(0)
            for i in range(obj.nVar+1):
                if f.item(i) >= 0:
                    xmin = xmin + f.item(i) * obj.predicate_lb[i-1]
                    xmax = xmax + f.item(i) * obj.predicate_ub[i-1]
                else:
                    xmin = xmin + f.item(i) * obj.predicate_ub[i-1]
                    xmax = xmax + f.item(i) * obj.predicate_lb[i-1]
        else:
            print('The ranges of predicate variables are unknown to estimate the ranges of the states, we solve LP optimization to get the exact range')
            [xmin, xmax] = obj.getRange(index)
        return [xmin, xmax]

    # estimate ranges using clip method from Stanley Bak
    # it is slower than the for-loop method
    def estimateRanges(obj):
        # return: lb: lower bound vector
        #         ub: upper bound vector

        pos_mat = np.where(obj.V > 0, obj.V, 0)
        neg_mat = np.where(obj.V < 0, obj.V, 0)

        xmin1 = pos_mat @ np.vstack((0, obj.predicate_lb))
        xmax1 = pos_mat @ np.vstack((0, obj.predicate_ub))
        xmin2 = neg_mat @ np.vstack((0, obj.predicate_ub))
        xmax2 = neg_mat @ np.vstack((0, obj.predicate_lb))

        lb = obj.V[:, 0] + xmin1 + xmin2
        ub = obj.V[:, 0] + xmax1 + xmax2
        return [lb, ub]

    # estimate range using clip
    # from Stanley Bak
    def estimateBound(obj, index):
        # @index: position of the state
        # range: min and max values of x[index]

        if index < 0 or index >= obj.dim:
            raise Exception('error: invalid index')

        f = obj.V[index, 1:obj.nVar + 1]

        pos_mat = np.matrix(np.where(f > 0, f, 0))
        neg_mat = np.matrix(np.where(f < 0, f, 0))

        xmin1 = pos_mat @ obj.predicate_lb
        xmax1 = pos_mat @ obj.predicate_ub
        xmin2 = neg_mat @ obj.predicate_ub
        xmax2 = neg_mat @ obj.predicate_lb

        xmin = obj.V[index, 0] + xmin1 + xmin2
        xmax = obj.V[index, 0] + xmax1 + xmax2
        return [xmin, xmax]

    # quickly estimate lower bound and upper bound vector of state variable
    def estimateBounds(obj):
        from engine.set.zono import Zono
        if isinstance(obj.Z, Zono):
            [lb, ub] = obj.Z.getBounds()
        else:
            n = obj.dim
            lb = np.zeros((n,1))
            ub = np.zeros((n,1))

            for i in range(n):
                [lb[i], ub[i]] = obj.estimateRange(i)
        return [lb, ub]

#------------------check if this function is working--------------------------------------------
    # check if an index is larger than other
    def is_p1_larger_than_p2(obj, p1_id, p2_id):
        # @p1_id: index of point 1
        # @p2_id: index of point 2
        # return = 1 if there exists the case that p1 >= p2
        #          2 if there is no case that p1 >= p2

        if p1_id < 0 or p1_id >= obj.dim:
            raise Exception('error: invalid index for point 1')

        if p2_id < 0 or p2_id >= obj.dim:
            raise Exception('error: invalid index for point 2')

        d1 = obj.V[p1_id, 0] - obj.V[p2_id, 0]
        C1 = obj.V[p2_id, 1:obj.nVar+1] - obj.V[p1_id, 1:obj.nVar+1]

        new_C = np.vstack((obj.C, C1))
        new_d = np.vstack((obj.d, d1))
        S = Star(obj.V, new_C, new_d, obj.predicate_lb, obj.predicate_ub)

        if S.isEmptySet():
            return 0
        else:
            return 1

#------------------check if this function is working--------------------------------------------
    # find a oriented box bound a star
    def getOrientedBox(obj):
        # !!! the sign of SVD result is different compared to Matalb
        # problem in shape=obj.dim => obj.nVar
        # bounds are correct but cannot getRanges
        [Q, sdiag, vh] = np.linalg.svd(obj.V[:, 1:obj.nVar + 1])
        Z = np.zeros((len(sdiag), obj.nVar))
        np.fill_diagonal(Z, sdiag)
        S = Z * vh
        print('Q: ', Q)
        print('Z: ', Z)
        print('S: ', S)
        print('P: ', vh)

        lb = np.zeros((obj.dim, 1))
        ub = np.zeros((obj.dim, 1))

        for i in range(obj.dim):
            f = S[i, :]

            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            x = min_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(obj.C)
            d = np.array(obj.d).flatten()
            min_.addConstr(C @ x <= d)
            min_.optimize()


            if min_.status == 2:
                lb[i] = min_.objVal
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            x = max_.addMVar(shape=obj.nVar, lb=obj.predicate_lb, ub=obj.predicate_ub)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(obj.C)
            d = np.array(obj.d).flatten()
            max_.addConstr(C @ x <= d)
            max_.optimize()

            if max_.status == 2:
                ub[i] = max_.objVal
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))

        print('lb: ', lb)
        print('ub: ', ub)
        new_V = np.hstack((obj.V[:, 0], Q))
        new_C = np.vstack((np.eye(obj.dim), -np.eye(obj.dim)))
        new_d = np.vstack((ub, -lb))
        return Star(new_V, new_C, new_d)

    def getZono(obj):
        from engine.set.box import Box
        B = obj.getBox()
        if isinstance(B, Box):
            return B.toZono()
        else:
            return np.array([])

    # def plot(obj):
    #     assert obj.dim <= 2 and obj.dim > 0, 'error: only 2D star can be plotted'

    #     P = pc.Polytope(obj.C, obj.d)
    # def plot(obj):
    #     # A = obj.V[:, 1:]
    #     A = np.vstack((obj.C, obj.V[:, 1:]))
    #     b = np.vstack((obj.d, obj.V[:, 0]))
    #     # halfspaces
    #     # H = np.hstack((obj.C, -obj.d))
    #     H = np.hstack((A, -b))
    #     print('H: \n', H)
    #     # feasible point
    #     p = np.array(obj.V[:, 0]).flatten()
    #     p = np.zeros(3)
    #     print('p: ', p)
    #     print('V: \n', obj.V)
    #     hs = scipy.spatial.HalfspaceIntersection(H, p)
    #     verts = hs.intersections
    #     hull = scipy.spatial.ConvexHull(verts)
    #     faces = hull.simplices

    #     ax = a3.Axes3D(plt.figure())
    #     ax.dist=10
    #     ax.azim=30
    #     ax.elev=10
    #     ax.set_xlim([-3,3])
    #     ax.set_ylim([-3,3])
    #     ax.set_zlim([-3,3])

    #     for s in faces:
    #         sq = [
    #             [verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
    #             [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
    #             [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]
    #         ]

    #         f = a3.art3d.Poly3DCollection([sq])
    #         f.set_color(colors.rgb2hex(scipy.rand(3)))
    #         f.set_edgecolor('k')
    #         f.set_alpha(0.1)
    #         ax.add_collection3d(f)
    #     plt.show()

    def __str__(obj):
        from engine.set.zono import Zono
        print('class: %s' % obj.__class__)
        print('V: [%sx%s %s]' % (obj.V.shape[0], obj.V.shape[1], obj.V.dtype))
        print('C: [%sx%s %s]' % (obj.C.shape[0], obj.C.shape[1], obj.C.dtype))
        print('dim: %s' % obj.dim)
        print('nVar: %s' % obj.nVar)
        if obj.predicate_lb.size:
            print('predicate_lb: [%sx%s %s]' % (obj.predicate_lb.shape[0], obj.predicate_lb.shape[1], obj.predicate_lb.dtype))
        else:
            print('predicate_lb: []')
        if obj.predicate_ub.size:
            print('predicate_ub: [%sx%s %s]' % (obj.predicate_ub.shape[0], obj.predicate_ub.shape[1], obj.predicate_ub.dtype))
        else:
            print('predicate_ub: []')
        if obj.state_lb.size:
            print('state_lb: [%sx%s %s]' % (obj.state_lb.shape[0], obj.state_lb.shape[1], obj.state_lb.dtype))
        else:
            print('state_lb: []')
        if obj.state_ub.size:
            print('state_ub: [%sx%s %s]' % (obj.state_ub.shape[0], obj.state_ub.shape[1], obj.state_ub.dtype))
        else:
            print('state_ub: []')
        if isinstance(obj.Z, Zono):
            print('Z: [1x1 Zono]')
        else:
            print('Z: []')
        return '\n'

    def __repr__(obj):
        from engine.set.zono import Zono
        if isinstance(obj.Z, Zono):
            return "class: %s \nV: %s \nC: %s \nd: %s \ndim: %s \nnVar: %s \npred_lb: %s \npred_ub: %s \nstate_lb: %s \nstate_ub: %s\nZ: %s" % (obj.__class__, obj.V, obj.C, obj.d, obj.dim, obj.nVar, obj.predicate_lb, obj.predicate_ub, obj.state_lb, obj.state_ub, obj.Z.__class__)
        return "class: %s \nV: %s \nC: %s \nd: %s \ndim: %s \nnVar: %s \npred_lb: %s \npred_ub: %s \nstate_lb: %s \nstate_ub: %s" % (obj.__class__, obj.V, obj.C, obj.d, obj.dim, obj.nVar, obj.predicate_lb, obj.predicate_ub, obj.state_lb, obj.state_ub)