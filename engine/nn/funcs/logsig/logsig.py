#!/usr/bin/python3
import numpy as np

import sys

sys.path.insert(0, "engine/set/box")
sys.path.insert(0, "engine/set/star")
from box import *
from star import *
class LogSig:
    """
        LogSig class contains method for reachability analysis for Layer with
        Sigmoid activation function.
        Reference: https://www.mathworks.com/help/deeplearning/ref/logsig.html
    """
    def evaluate(x):
        return 1 / (1 + np.exp(-x))     # np.exp(-np.logaddexp(0, -x)) check if it is more accurate

    def logsig(x):
        return 1 / (1 + np.exp(-x))
    
    def dlogsig(x):
        """ 
            Derivative of logsig(x)
        """
        f = LogSig.logsig(x)
        return f * (1 - f)
        
    # main method
    def reach_star_approx(I,                                # input star set
                        method = 'approx-star-no-split',    # reach star approximate method    
                        relaxFactor = 0,                    # for relaxed approx-star method
                        disp_opt = '',                      # display option
                        lp_solver = 'gurobi'):              # lp solver option
        from star import Star
        
        assert isinstance(I, Star), 'error: input set is not a star set'

        if method == 'approx-star-no-split' or method == 'approx-star':
            if relaxFactor == 0:
                S = LogSig.multiStepLogSig_NoSplit(I, disp_opt, lp_solver)
            else:
                S = LogSig.relaxedMultiStepLogSig_NoSplit(I, relaxFactor, disp_opt, lp_solver)
        elif method == 'approx-star-split':
            S = LogSig.reach_star_approx_split(I)
        else:
            raise Exception('error: unkown reachability method')
        return S

#------------------check if this function is working--------------------------------------------
    # reachability method with star
    # def reach_star_approx_split(I):

 #------------------check if this function is working--------------------------------------------
    # def stepLogSig_Split(I, index):

    # multiStepLogSig at one
    def multiStepLogSig_NoSplit(I, disp_opt = '', lp_solver = 'gurobi'):
        # @I: input star set
        # @l: l = min(x[index]), lower bound at neuron x[index]
        # @u: u = max(x[index]), upper bound at neuron x[index]
        # @yl: yl = tansig(l), output of logsig at lower bound
        # @yu: yu = tansig(u), output of logsig at upper bound
        # @dyl: derivative of TanSig at the lower bound
        # @dyu: derivative of TanSig at the upper bound
        # return: output star set
        from star import Star

        assert isinstance(I, Star), 'error: input set is not a star'

        N = I.dim
        inds = np.array(range(N))
        disp = disp_opt == 'display'
        if disp:
            print('\nComputing lower-bounds: ')
        l = I.getMins(inds, '', disp_opt, lp_solver)
        if disp: 
            print('\nComputing upper-bounds: ')
        u = I.getMaxs(inds, '', disp_opt, lp_solver)
    
        yl = LogSig.logsig(l)
        yu = LogSig.logsig(u)
        dyl = LogSig.dlogsig(l)
        dyu = LogSig.dlogsig(u)

        # l ~= u
        map2 = np.argwhere(l != u)
        m = len(map2)
        V2 = np.zeros([N, m])
        for i in range(m):
            V2[map2[i], i] = 1

        # new basis matrix
        new_V = np.hstack([np.zeros([N, I.nVar+1]), V2])
    
        # l == u
        map1 = np.argwhere(l == u)
        if len(map1):
            yl1 = yl[map1]
            new_V[map1, 0] = yl1
            new_V[map1, 1:I.nVar+1+m] = 0

        # add new constraints
        
        # C0, d0
        n = I.C.shape[0]
        C0 = np.hstack([I.C, np.zeros([n,m])])
        d0 = I.d

        nv = I.nVar + 1

        # C1, d1, x >= 0
        # constraint 1: y <= y'(l) * (x - l) + y(l)
        # constarint 2: y <= y'(u) * (x - u) + y(u) 
        # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l)
        map1 = np.argwhere((l >= 0) & (l != u)).flatten()
        if len(map1):
            a = yl[map1]
            b = yu[map1]
            da = dyl[map1]
            db = dyu[map1]
            # constraint 1: y <= y'(l) * (x - l) + y(l)
            C11 = np.hstack([-da * I.V[map1, 1:nv], V2[map1, :]])
            d11 = da * (I.V[map1, 0] - l[map1]) + a
            # constarint 2: y <= y'(u) * (x - u) + y(u)
            C12 = np.hstack([-db * I.V[map1, 1:nv], V2[map1, :]])
            d12 = db * (I.V[map1, 0] - u[map1]) + b
            # constraint 3: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l)
            gamma = (b-a)/(u[map1] - l[map1])
            C13 = np.hstack([gamma * I.V[map1, 1:nv], -V2[map1, :]])
            d13 = -gamma * (I.V[map1, 0] - l[map1]) - a

            C1 = np.vstack([C11, C12, C13])
            d1 = np.hstack([d11, d12, d13])
        else:
            C1 = np.empty([0, nv+1])
            d1 = np.empty(0)

        # C2, d2, x <= 0
        # y is concave when x <= 0
        # constraint 1: y >= y'(l) * (x - l) + y(l)
        # constraint 2: y >= y'(u) * (x - u) + y(u)
        # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l)
        map1 = np.argwhere((u <= 0) & (l != u)).flatten()
        if len(map1):
            a = yl[map1]
            b = yu[map1]
            da = dyl[map1]
            db = dyu[map1]
            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C21 = np.hstack([da * I.V[map1, 1:nv], -V2[map1, :]])
            d21 = -da * (I.V[map1, 0] - l[map1]) - a
            # constraint 2: y >= y'(u) * (x - u) + y(u)
            C22 = np.hstack([db * I.V[map1, 1:nv], -V2[map1, :]])
            d22 = -db * (I.V[map1, 0] - u[map1]) - b
            # constraint 3: y <= (y(u) - y(l)) * (x -l) / (u - l) + y(l)
            gamma = (b-a)/(u[map1] - l[map1])
            C23 = np.hstack([-gamma * I.V[map1, 1:nv], V2[map1, :]])
            d23 = gamma * (I.V[map1, 0] - l[map1]) + a

            C2 = np.vstack([C21, C22, C23])
            d2 = np.hstack([d21, d22, d23])
        else:
            C2 = np.empty([0, nv+1])
            d2 = np.empty(0)

        # C3, d3, l< 0 and u > 0, x > 0 or x < 0
        # y is concave for x in [l, 0] and convex for x
        # in [0, u]
        # split can be done here
        map1 = np.argwhere((l < 0) & (u > 0)).flatten()
        if len(map1):
            a = yl[map1]
            b = yu[map1]
            da = dyl[map1]
            db = dyu[map1]

            dmin = np.minimum(da, db)
            # over-approximation constraints
            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)
            # constraint 3: y <= g2 * x + y2
            # constraint 4: y >= g1 * x + y1

            # constraint 1: y >= min(y'(l), y'(u)) * (x - l) + y(l)
            C31 = np.hstack([dmin * I.V[map1, 1:nv], -V2[map1, :]])
            d31 = -dmin * (I.V[map1, 0] - l[map1]) - a
            # constraint 2: y <= min(y'(l), y'(u)) * (x - u) + y(u)
            C32 = np.hstack([-dmin * I.V[map1, 1:nv], V2[map1, :]])
            d32 = dmin * (I.V[map1, 0] - u[map1]) + b

            # dmin = np.minimum(da, db)
            # y1 = dmin*(-l[map1]) + a
            # y2 = dmin*(-u[map1]) + b
            # g2 = (y2 - a)/(-l[map1])
            # g1 = (y1 - b)/(-u[map1])
            
            # # constraint 3: y <= g2 * x + y2
            # C33 = np.hstack((np.multiply(-g2,I.V[map1, 1:nv]), V2[map1, :]))
            # d33 = np.multiply(g2, I.V[map1, 0]) + y2
            # # constraint 4: y >= g1 * x + y1
            # C34 = np.hstack((np.multiply(g1,I.V[map1, 1:nv]), -V2[map1, :]))
            # d34 = np.multiply(-g1, I.V[map1, 0]) - y1

            l_map = l[map1]
            u_map = u[map1]
            y0 = LogSig.logsig(0)
            dy0 = LogSig.dlogsig(0)
            gu_x = (b - dmin * u_map - y0) / (dy0 - dmin)
            gu_y = dy0 * gu_x + y0
            gl_x = (a - dmin * l_map - y0) / (dy0 - dmin)
            gl_y = dy0 * gl_x + y0
            
            mu = (a - gu_y) / (l_map - gu_x) 
            ml = (b - gl_y) / (u_map - gl_x)

            # constraint 3: y[index] >= m_l * x[index] - m_l*u + y_u
            C33 = np.hstack([ml * I.V[map1, 1:nv], -V2[map1, :]])
            d33 = -ml * I.V[map1, 0] + ml * u_map - b
            
            # constraint 4: y[index] <= m_u * x[index] - m_u*l + y_l
            C34 = np.hstack([-mu * I.V[map1, 1:nv], V2[map1, :]])
            d34 = mu * I.V[map1, 0] - mu * l_map + a
            
            C3 = np.vstack([C31, C32, C33, C34])
            d3 = np.hstack([d31, d32, d33, d34])
        else:
            C3 = np.empty([0, nv+1])
            d3 = np.empty(0)
            
        new_C = np.vstack([C0, C1, C2, C3])
        new_d = np.hstack([d0, d1, d2, d3])
        
        new_pred_lb = np.hstack([I.predicate_lb, yl[map2].flatten()])
        new_pred_ub = np.hstack([I.predicate_ub, yu[map2].flatten()])

        return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub)

#------------------check if this function is working--------------------------------------------
    # def relaxedMultiStepLogSig_NoSplit(I, relaxFactor = 0, disp_opt = '', lp_solver = 'gurobi'):
    #     # @I: input star set
    #     # @relaxFactor: for relaxed approx-star method
    #     # @dis_opt; display option = '' or 'display'

    #     # @l: l = min(x[index]), lower bound at neuron x[index]
    #     # @u: u = min(x[index]), upper bound at neuron x[index]
    #     # @yl: = logsig(l); output of logsig at lower bound
    #     # @yu: = logsig(u); output of logsig at upper bound
    #     # @dyl: derivative of LogSig at the lower bound
    #     # @dyu: derivative of LogSig at the upper bound


    def reach_rstar_approx(I):
        # @I: input RStar set
        # return: output RStar set
        from rstar import RStar
        
        assert isinstance(I, RStar), 'error: input set is not a RStar set'

        # LogSig.multiLogSig_rstar(I)

        D_L = I.D_L
        D_U = I.D_U
        lb = I.lb
        ub = I.ub
        n = len(D_L) - 1

        l = lb[n]
        u = ub[n]

        y_l = LogSig.evaluate(l)
        y_u = LogSig.evaluate(u)
        dy_l = (y_l * (1 - y_l.T)).diagonal().reshape(-1,1)
        dy_u = (y_u * (1 - y_u.T)).diagonal().reshape(-1,1)

        # create new matrices for lower and uppper polyhedral constraints and bounds
        D_L.append(np.zeros((I.dim, I.dim + 1)))
        D_U.append(np.zeros((I.dim, I.dim + 1)))
        lb.append(np.zeros((I.dim, 1)))
        ub.append(np.zeros((I.dim, 1)))
        RS = RStar(I.V, I.C, I.d, I.predicate_lb, I.predicate_ub, D_L, D_U, lb, ub, I.iter)

        for i in range(I.dim):
            RS = LogSig.stepLogSig_rstar(RS, i, l[i], u[i], y_l[i], y_u[i], dy_l[i], dy_u[i])
        return RS

    def stepLogSig_rstar(I, index, l, u, y_l, y_u, dy_l, dy_u):
        # @I: rstar-input set
        # @index: index of neuron performing stepReach
        # @l: l = min(x[index]), lower bound at neuron x[index] 
        # @u: u = max(x[index]), upper bound at neuron x[index]
        # @y_l: = logsig(l); output of logsig at lower bound
        # @y_u: = logsig(u); output of logsig at upper bound
        # @dy_l: derivative of LogSig at the lower bound
        # @dy_u: derivative of LogSig at the upper bound
        # return: RStar output set
        from rstar import RStar

        assert isinstance(I, RStar), 'error: input set is not a RStar set'
        
        D_L = I.D_L
        D_U = I.D_U
        lb = I.lb
        ub = I.ub
        n = len(D_L) - 1

        if l == u:
            new_V = I.V
            new_V[index, :] = 0
            new_V[index, 0] = y_l

            L = np.zeros((1, I.dim + 1))
            L[index + 1] = y_l
            D_L[n][index, :] = L

            U = np.zeros((1, I.dim + 1))
            U[index + 1] = y_l
            D_U[n][index, :] = U

            lb[n][index] = y_l
            ub[n][index] = y_u

            return RStar(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, D_L, D_U, lb, ub, I.iter)
        else:
            # print('I: \n', I)
            # print('I.V: \n', I.V)
            # print(' np.zeros((I.dim, 1)): \n',  np.zeros((I.dim, 1)))
            # print('I.C: \n', I.C)
            # print('np.zeros((I.C.shape[0], 1)): \n', np.zeros((I.C.shape[0], 1)))
            new_V = np.column_stack((I.V, np.zeros((I.dim, 1))))
            new_V[index, :] = 0
            new_V[index, -1] = 1

            C0 = np.column_stack((I.C, np.zeros((I.C.shape[0], 1))))
            d0 = I.d

            if l >= 0:
                a = (y_u - y_l)/(u - l)

                # constraint 1: y[index] >= y(l) + a * (x[index] - l)
                L = np.zeros(I.dim + 1).T
                L[0] = y_l - a * l
                L[index + 1] = a
                D_L[n][index, :] = L

                C1 = np.column_stack((a*I.V[index, 1:], -1))
                d1 = -y_l - a * (I.V[index, 0] - l)

                # constraint 2: y[index] <= y(u) + y'(u) * (x[index] - u)
                U = np.zeros(I.dim + 1).T
                U[0] = y_l - dy_u * u
                U[index + 1] = dy_u
                D_U[n][index, :] = U

                C2 = np.column_stack((-dy_u*I.V[index, 1:], 1))
                d2 = y_u + dy_u * (I.V[index, 0] - u)
            elif u <= 0:
                a = (y_u - y_l)/(u - l)

                # constraint 1: y[index] >= y(l) + y'(l) * (x[index] - l)
                L = np.zeros(I.dim + 1).T
                L[0] = y_l - dy_l * l
                L[index + 1] = dy_l
                D_L[n][index, :] = L

                C1 = np.column_stack((dy_l*I.V[index, 1:], -1))
                d1 = -y_l - dy_l * (I.V[index, 0] - l)

                # constraint 2: y[index] <= y(l) + a * (x[index] - l)
                U = np.zeros(I.dim + 1).T
                U[0] = y_l - a * l
                U[index + 1] = a
                D_U[n][index, :] = U
                
                C2 = np.column_stack((-a * I.V[index, 1:], 1))
                d2 = y_l + a * (I.V[index, 0] - l)
            else:
                da = min(dy_l, dy_u)

                # constraint 1: y[index] >= y(l) + da * (x[index] - l)
                L = np.zeros(I.dim + 1).T
                L[0] = y_l - da * l
                L[index + 1] = da
                D_L[n][index, :] = L
                
                C1 = np.column_stack((da*I.V[index, 1:], -1))
                d1 = -y_l - da * (I.V[index, 0] - l)

                # constraint 2: y[index] <= y(u) + lamda' * (x[index] - u)
                U = np.zeros(I.dim + 1).T
                U[0] = y_u - da * u
                U[index + 1] = da
                D_U[n][index, :] = U

                C2 = np.column_stack((-da*I.V[index, 1:], 1))
                d2 = y_u + da * (I.V[index, 0] - u)


            lb[n][index] = y_l
            ub[n][index] = y_u

            new_C = np.row_stack((C0, C1, C2))
            new_d = np.row_stack((d0, d1, d2))

            new_pred_lb = np.vstack((I.predicate_lb, y_l)) 
            new_pred_ub = np.vstack((I.predicate_ub, y_u))

            return RStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub, D_L, D_U, lb, ub, I.iter)

    # def multistepLogSig_rstar(I):
    #     # @I: input RStar set
    #     # return: output RStar set
    #     from engine.set.rstar import RStar
        
    #     assert isinstance(I, RStar), 'error: input set is not a RStar set'

    #     N = I.dim
    #     nVar = I.V.shape[1]

    #     D_L = I.D_L
    #     D_U = I.D_U
    #     lb = I.lb
    #     ub = I.ub
    #     n = len(D_L) - 1

    #     l = lb[n]
    #     u = ub[n]

    #     y_l = LogSig.evaluate(l)
    #     y_u = LogSig.evaluate(u)
    #     dy_l = (y_l * (1 - y_l.T)).diagonal().reshape(-1,1)
    #     dy_u = (y_u * (1 - y_u.T)).diagonal().reshape(-1,1)

    #     # create new matrices for lower and uppper polyhedral constraints and bounds
    #     D_L.append(np.zeros((I.dim, I.dim + 1)))
    #     D_U.append(np.zeros((I.dim, I.dim + 1)))
    #     lb.append(np.zeros((I.dim, 1)))
    #     ub.append(np.zeros((I.dim, 1)))

    #     # l ~= u
    #     map2 = np.argwhere(l.flatten() != u.flatten())
    #     m = len(map2)
    #     V2 = np.zeros((N,m))
    #     for i in range(m):
    #         V2[map2[i], i] = 1

    #     # new basis matrix
    #     new_V = np.hstack((np.zeros((N, I.nVar+1)), V2))

    #     # l == u
    #     map1 = np.argwhere(l.flatten() == u.flatten())
    #     if len(map1):
    #         yl1 = y_l[map1]
    #         new_V[map1, 0] = yl1
    #         new_V[map1, 1:nVar+1+m] = 0


    #     # add new constraints
        
    #     # C0, d0
    #     n = I.C.shape[0]
    #     C0 = np.hstack((I.C, np.zeros((n,m))))
    #     d0 = I.d

    #     nv = nVar+1

    #     # C1, d1, x >= 0
    #     # constraint 1: y >= y(l) + a * (x- l)
    #     # constraint 2: y <= y(u) + y'(u) * (x- u)
    #     map1 = np.argwhere((l.flatten() >= 0) & (l.flatten() != u.flatten()))
    #     if len(map1):



    #         a = (y_u - y_l)/(u - l)

    #         # constraint 1: y[index] >= y(l) + a * (x[index] - l)
    #         L = np.zeros(I.dim + 1).T
    #         L[0] = y_l - a * l
    #         L[index + 1] = a
    #         D_L[n][index, :] = L

    #         C1 = np.column_stack((a*I.V[index, 1:], -1))
    #         d1 = -y_l - a * (I.V[index, 0] - l)

    #         # constraint 2: y[index] <= y(u) + y'(u) * (x[index] - u)
    #         U = np.zeros(I.dim + 1).T
    #         U[0] = y_l - dy_u * u
    #         U[index + 1] = dy_u
    #         D_U[n][index, :] = U

    #         C2 = np.column_stack((-dy_u*I.V[index, 1:], 1))
    #         d2 = y_u + dy_u * (I.V[index, 0] - u)    
    #     else:
    #         C1 = np.empty((0, nv+1))
    #         d1 = np.empty((0, 1))

    #     # C2, d2, x <= 0


    #     # C3, d3, l< 0 and u > 0, x > 0 or x < 0


    #     RS = RStar(I.V, I.C, I.d, D_L, D_U, lb, ub, I.iter)

    #     for i in range(I.dim):
    #         RS = LogSig.stepLogSig_rstar(RS, i, l[i], u[i], y_l[i], y_u[i], dy_l[i], dy_u[i])
    #     return RS



    #     return

    #-------------------------------- over-approximate reachability analysis with zonotope -----------------------------------#
    # rechability analysis with zonotope
    def reach_zono_approx(I):
        """
            Rechability analysis with zonotope.
            Approximates logistic sigmoid function by a zonotope.
            I: input zono
            
            return: Z -> output zono
            
            reference: Fast and Effective Robustness Certification,
                       Gagandeep Singh, NIPS, 2018
        """
        from zono import Zono

        assert isinstance(I, Zono), 'error: input set is not a Zono'

        B = I.getBox()
        lb = B.lb
        ub = B.ub
        
        y_lb = LogSig.logsig(lb)
        y_ub = LogSig.logsig(ub)

        G = np.vstack([LogSig.dlogsig(lb), LogSig.dlogsig(ub)])
        gamma_opt = np.min(G, axis = 0)
        gamma_mat = np.diag(gamma_opt)
        mu1 = 0.5 * (y_ub + y_lb - gamma_mat @ (ub + lb))
        mu2 = 0.5 * (y_ub - y_lb - gamma_mat @ (ub - lb))
        Z1 = I.affineMap(gamma_mat, mu1)
        new_V = np.diagflat(mu2)

        V = np.hstack([Z1.V, new_V])
        return Zono(Z1.c, V)

#------------------check if this function is working--------------------------------------------
    # dealing with multiple inputs in parallel
    # def reach_zono_approx_multipleInputs(I, parallel = ''):


    #-------------------------------- over-approximate reachability analysis with abstract domain -----------------------------------#
    def stepLogSig_absdom(I, index, l, u, y_l, y_u, dy_l, dy_u):
        # @I: input star set
        # @index: index of the neuron
        # @l:     l = min(x[index]); lower bound at neuron x[index]
        # @u:     u = min(x[index]); upper bound at neuron x[index]
        # @y_l: y_l = logsig(l); output of logsig at lower bound
        # @y_u: y_u = logsig(u); output of logsig at upper bound
        # return: output star set
        from star import Star

        if l == u:
            new_V = I.V
            new_V[index, :] = 0
            new_V[index, 0] = y_l
            return Star(new_V, I.C, I.predicate_lb, I.predicate_ub)

        elif l >= 0:
            # y is convex when x >= 0
            # constraint 1: y <= y'(u) * (x - u) + y(u)
            # constraint 2: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l)

            n = I.nVar + 1
            # over-approximation constraints
            # constraint 1: y <= y'(u) * (x - u) + y(u)
            C1 = np.hstack([-dy_u*I.V[index, 1:n], 1])
            d1 = dy_u * I.V[index, 0] - dy_u*u + y_u
            # constraint 2: y >= (y(u) - y(l)) * (x - l) / (u - l) + y(l)
            a = (y_u - y_l) / (u - l)
            C2 = np.hstack([a*I.V[index, 1:n], -1])
            d2 = a*l - y_l - a*I.V[index, 0]

            m = I.C.shape[0]
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            d0 = I.d
            new_C = np.vstack([C0, C1, C2])
            new_d = np.hstack([d0, d1, d2])
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            new_V[index, :] = 0
            new_V[index, n] = 1

            # update predicate bound
            new_predicate_lb = np.hstack([I.predicate_lb, y_l])
            new_predicate_ub = np.hstack([I.predicate_ub, y_u])
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)
        
        elif u <= 0:
            # y is concave when x <= 0
            # constraint 1: y >= y'(l) * (x - l) + y(l)
            # constraint 2: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l)

            n = I.nVar + 1
            # over-approximation constraints
            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C1 = np.hstack([dy_l*I.V[index, 1:n], -1])
            d1 = -dy_l * I.V[index, 0] + dy_l*l - y_l
            # constraint 2: y <= (y(u) - y(l)) * (x - l) / (u - l) + y(l)
            a = (y_u - y_l)/(u - l)
            C2 = np.hstack([-a*I.V[index, 1:n], 1]) 
            d2 = -a*l + y_l + a*I.V[index, 0]

            m = I.C.shape[0]
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            d0 = I.d
            new_C = np.vstack([C0, C1, C2])
            new_d = np.hstack([d0, d1, d2])
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            new_V[index, :] = 0
            new_V[index, n] = 1

            # update predicate bound
            new_predicate_lb = np.hstack([I.predicate_lb, y_l])
            new_predicate_ub = np.hstack([I.predicate_ub, y_u])
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)

        elif l < 0 and u > 0:
            # over-approximation constraints 
            # constraint 1: y >= y'(l) * (x - l) + y(l)
            # constraint 2: y <= y'(u) * (x - u) + y(u)

            n = I.nVar + 1

            dy_min = min(dy_l, dy_u)
            # constraint 1: y >= y'(l) * (x - l) + y(l)
            C1 = np.hstack([dy_min*I.V[index, 1:n], -1])
            d1 = -dy_min * I.V[index, 0] + dy_min*l - y_l
            # constraint 2: y <= y'(u) * (x - u) + y(u)
            C2 = np.hstack([-dy_min*I.V[index, 1:n], 1])
            d2 = dy_min * I.V[index, 0] - dy_min*u + y_u

            m = I.C.shape[0]
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            d0 = I.d
            new_C = np.vstack([C0, C1, C2])
            new_d = np.hstack([d0, d1, d2])
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            new_V[index, :] = 0
            new_V[index, n] = 1

            # update predicate bound
            new_predicate_lb = np.hstack([I.predicate_lb, y_l])
            new_predicate_ub = np.hstack([I.predicate_ub, y_u])
            return Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub)

    # reachability analysis with abstract domain
    def reach_absdom_approx(I):
        # @I: input star set
        # return: soutput star set

        # reference: An abstract domain for certifying neural networks. Proceedings of the ACM on Programming Languages,
        # Gagandeep Singh, POPL, 2019
        from star import Star

        assert isinstance(I, Star), 'error: input set is not a Star'

        [l, u] = I.estimateRanges()

        y_l = LogSig.logsig(l)
        y_u = LogSig.logsig(u)
        
        dy_l = LogSig.dlogsig(l)
        dy_u = LogSig.dlogsig(u)

        n = I.dim
        S = I
        for i in range(n):
            S = LogSig.stepLogSig_absdom(S, i, l[i], u[i], y_l[i], y_u[i], dy_l[i], dy_u[i])
        return S

    # main function for reachability analysis
    def reach(I,             # an input star set
            method,          # 'approx-star', 'approx-zono', or 'abs-dom'
            option = '',     # = 'parallel' or '' using parallel computation or not
            relaxFactor = 0, # for relaxed approx-star method 
            disp_opt = '', 
            lp_solver = 'gurobi'):

        if method == 'approx-star':     # exact analysis using star
            return LogSig.reach_star_approx(I, method, relaxFactor, disp_opt, lp_solver)
        elif method == 'approx-rstar':
            return LogSig.reach_rstar_approx(I)
        elif method == 'approx-zono':   # over-approximate analysis using zonotope
            return LogSig.reach_zono_approx(I)
        elif method == 'abs-dom':       # over-approximate analysis using abstrac-domain method of star
            return LogSig.reach_absdom_approx(I)
        else:
            raise Exception('error: unknown or unsupported reachability method for layer with TanSig activation function')