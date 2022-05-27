#!/usr/bin/python3

import copy
import numpy as np
# from sklearn import preprocessing
# from numba import njit, prange

# preprocessing library which contains the normalize function to normalize
# the data. It takes an array in as an input and normalizes its values
# between 00 and 11. It then returns an output array with the same
# dimensions as the input.

class PosLin:
    # PosLin class contains method for reachability analysis for Layer with
    # ReLU activation function (ReLU operator in NN)

    def evaluate(x): # evaluate method and reachability analysis with stars
        # n = len(x)
        # if (len(x)[0] != 1):
        #     'error: x is not a vector'
        # y = np.zeros(n, 1)
        # for i in range(1,n):
        #     if x[i][1] < 0:
        #         y[i][1] == 0
        #     else:
        #         y[i][1] = x[i][1]
        # return y
        return np.maximum((x, 0))

    # stepReach method, compute reachable set for a single step
    def stepReach(*args):
        # @I, input set, a star (array of a single star)
        # @index, index of current x[index] of current step,
        # should be the number from matlab - 1
        # @xmin, min of x[index]
        # @xmax, max of x[index]
        # @S, star output set

        #print(len(args))
        if (len(args) == 2): # 2 arguments
            I = args[0]
            index = args[1]
            lp_solver = 'gurobi'
        elif (len(args) == 3): # 3 arguments
            I = args[0]
            index = args[1]
            lp_solver = args[2]
        else: 'error: Invalid number of input arguments, should be 2 or 3'

        from engine.set.star import Star

        assert isinstance(I, Star), 'error: input set is not a star set'

        # need to fix gurobi problem
        #xmin = I.getMin(index, lp_solver)
        [xmin, xmax] = I.estimateRange(index)

        from engine.set.zono import Zono

        if xmin >= 0:
            S = I
            return S
        else:
            #xmax = I.getMax(index, lp_solver)

            if xmax <= 0:
                V1 = copy.deepcopy(I.V)
                V1[index, :] = 0
                if not isinstance(I.Z, Zono):
                    c = copy.deepcopy(I.Z.c)
                    c[index] = 0
                    V = copy.deepcopy(I.Z.V)
                    V[index, :] = 0
                    new_Z = Zono(c, V)
                else:
                    new_Z = np.array([])
                S = Star(V=V1, C=I.C, d=I.d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
                return S
            else:
                # S1 = I && x[index] < 0
                c = copy.deepcopy(I.V[index, 0])
                V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
                new_C = np.vstack([I.C, V])
                new_d = np.vstack([I.d, -c])
                new_V = copy.deepcopy(I.V)
                new_V[index, :] = np.zeros([1, I.nVar + 1])

                # update outer-zono
                if isinstance(I.Z, Zono):
                    c1 = copy.deepcopy(I.Z.c)
                    c1[index] = 0
                    V1 = copy.deepcopy(I.Z.V)
                    V1[index, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = np.array([])
                S1 = Star(V=new_V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)

                # S2 = I && x[index] >= 0
                new_C1 = np.vstack([I.C, -V])
                new_d1 = np.vstack([I.d, c])

                # S2 = Star(V=I.V, C=new_C1, d=new_d1, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=I.Z)
                S2 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub, outer_zono=I.Z)

                #S = np.column_stack([S1, S2])
                S = []
                S.append(S1)
                S.append(S2)
                return S

    # # new stepReach method, compute reachable set for a single step
    # # minimize the number of LP using simulation
    # def stepReach2(I, index):
    #     # @I: single star set input
    #     # @index: index of the neuron performing stepPosLin
    #     # @xmin: minimum of x[index]
    #     # @xmax: maximum mof x[index]
    #     # @S: star output set
    #
    #     from engine.set.star import Star
    #     from engine.set.zono import Zono
    #
    #     assert isinstance(I, Star), 'error: input is not a star set'
    #
    #     x1 = I.V[index, 0] + I.V[index, 1:I.nVar + 1] * I.predicate_lb
    #     x2 = I.V[index, 0] + I.V[index, 1:I.nVar + 1] * I.predicate_ub
    #
    #     if x1*x2 < 0:
    #         # S1 = I && x[index] < 0
    #         c = copy.deepcopy(I.V[index, 0])
    #         V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
    #         new_C = np.vstack([I.C, V])
    #         new_d = np.vstack([I.d, -c])
    #         new_V = copy.deepcopy(I.V)
    #         new_V[index, :] = np.zeros([1, I.nVar + 1])
    #
    #         # update outer-zono
    #         if not isinstance(I.Z, Zono):
    #             c1 = copy.deepcopy(I.Z.c)
    #             c1[index] = 0
    #             V1 = copy.deepcopy(I.Z.V)
    #             V1[index, :] = 0;
    #             new_Z = Zono(c1, V1)
    #         else:
    #             new_Z = np.array([])
    #         S1 = Star(V=new_V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
    #
    #         # S2 = I && x[index] >= 0
    #         new_C = np.vstack([I.C, -V])
    #         new_d = np.vstack([I.d, c])
    #         S2 = Star(V=I.V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=I.Z)
    #
    #         S = np.column_stack([S1, S2])
    #         return S
    #
    #     else:
    #         if (x1 < 0 and x2 < 0):
    #             xmax = I.getMax(index)
    #             if xmax <= 0:
    #                 V1 = copy.deepcopy(I.V)
    #                 V1[index, :] = 0
    #                 if not isinstance(I.Z, Zono):
    #                     c = copy.deepcopy(I.Z.c)
    #                     c[index] = 0
    #                     V = copy.deepcopy(I.Z.V)
    #                     V[index, :] = 0
    #                     new_Z = Zono(c, V) # update outer-zono
    #                 else:
    #                     new_Z = np.array([])
    #                 S = Star(V=V1, C=I.C, d=I.d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
    #                 return S
    #             else:
    #                 # S1 = I && x[index] < 0
    #                 c = copy.deepcopy(I.V[index, 0])
    #                 V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
    #                 new_C = np.vstack([I.C, V])
    #                 new_d = np.vstack([I.d, -c])
    #                 new_V = copy.deepcopy(I.V)
    #                 new_V[index, :] = np.zeros([1, I.nVar + 1])
    #
    #                 # update outer-zono
    #                 if not isinstance(I.Z, Zono):
    #                     c1 = copy.deepcopy(I.Z.c)
    #                     c1[index] = 0
    #                     V1 = copy.deepcopy(I.Z.V)
    #                     V1[index, :] = 0
    #                     new_Z = Zono(c1, V1)
    #                 else:
    #                     new_Z = np.array([])
    #                 S1 = Star(V=new_V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
    #
    #                 # S2 = I && x[index] >= 0
    #                 new_C = np.vstack([I.C, -V])
    #                 new_d = np.vstack([I.d, c])
    #
    #                 S2 = Star(V=I.V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=I.Z)
    #
    #                 S = np.column_stack([S1, S2])
    #                 return S
    #         else:
    #             xmin = I.getMin(index)
    #             if xmin >= 0:
    #                 S = I
    #                 return S
    #             else:
    #                 # S1 = I && x[index] < 0
    #                 c = copy.deepcopy(I.V[index, 0])
    #                 V = copy.deepcopy(I.V[index, 1:I.nVar + 1])
    #                 new_C = np.vstack([I.C, V])
    #                 new_d = np.vstack([I.d, -c])
    #                 new_V = copy.deepcopy(I.V)
    #                 new_V[index, :] = np.zeros([1, I.nVar + 1])
    #
    #                 # update outer-zono
    #                 if not isinstance(I.Z, Zono):
    #                     c1 = copy.deepcopy(I.Z.c)
    #                     c1[index] = 0
    #                     V1 = copy.deepcopy(I.Z.V)
    #                     V1[index, :] = 0
    #                     new_Z = Zono(c1, V1)
    #                 else:
    #                     new_Z = np.array([])
    #                 S1 = Star(V=new_V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
    #
    #                 # S2 = I && x[index] >= 0
    #                 new_C = np.vstack([I.C, -V])
    #                 new_d = np.vstack([I.d, c])
    #                 S2 = Star(V=I.V, C=new_C, d=new_d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=I.Z)
    #
    #                 S = np.column_stack([S1, S2])
    #                 return S

    # stepReach with multiple inputs
    def stepReachMultipleInputs(*args):
        # @I: an list of stars
        # @index: index where stepReach is performed
        # @option: = 'parallel' use parallel computing
        #          = not declare -> don't use parallel computing

        if (len(args) == 3): # 3 arguments
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = 'gurobi'
        elif (len(args) == 4): # 4 arguments
            I = args[0]
            index = args[1]
            option = args[2]
            lp_solver = args[3]
        else: 'error: Invalid number of input arguments'

        from engine.set.star import Star

        assert isinstance(I, list), 'error: input is not an array star sets'
        assert isinstance(I[0], Star), 'error: input is not a star set'

        #S = np.array([])
        # S1 = []
        # S1 = PosLin.stepReach(I[0], index, lp_solver)
        # print('S1 ---------', S1)

        p = len(I)
        S = []

        if len(option) - 1 == 0:
            for i in range(p):
                S1 = PosLin.stepReach(I[i], index, lp_solver)
                # print('\n S1 ------------ \n', S1)
                # S = np.array([S, S1]) if S.size else S1
                if len(S):
                   S.extend(S)
                   S.extend(S1)
                   del S[:2]
                else:
                   S.extend(S1)
        # elif option == 'parellel':
        #     #@njit(parallel=True)
        #     for i in prange(p):
        #         S = np.column_stack([S, PosLin.stepReach(I[i], index, lp_solver)])
        else: 'error: Unknown option'

        return S

    # exact reachability analysis using star
    def reach_star_exact(*args):
        # @I: star input sets
        # @option: = 'parallel' use parallel computing
        #          = not declare -> don't use parallel computing
        if len(args) == 2: # 2 arguments
            I = args[0]
            option = args[1]
            dis_opt = ''
            lp_solver = 'gurobi'
        elif len(args) == 3: # 3 arguments
            I = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = 'gurobi'
        elif len(args) == 4: # 4 arguments
            I = args[0]
            option = args[1]
            dis_opt = args[2]
            lp_solver = args[3]
        else: 'error: Invalid number of input arguments, should be 2, 3 or 4'

        from engine.set.star import Star
        from engine.set.zono import Zono

        # need to fix the gurobi problem
        #if not Star.isEmptySet(I):
        if isinstance(I, Star):
            [lb, ub] = Star.estimateRanges(I)

            if len(lb) == 0 or len(ub) == 0:
                # S = np.array([])
                S = []
                return S
            else:
                flatten_ub = np.ndarray.flatten(ub, 'F')
                map = np.argwhere(flatten_ub <= 0)
                ub_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_ub[0]) + map[0][i]
                    index = map[i][1]
                    ub_map = np.append(ub_map, index)
                V = copy.deepcopy(I.V)
                ### V[ub_map, :] = 0

                # update outer-zono
                if isinstance(I.Z, Zono):
                    c1 = copy.deepcopy(I.Z.c)
                    #c1[map, :] = 0
                    V1 = copy.deepcopy(I.Z.V)
                    #V1[map, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = np.array([])

                In = Star(V=V, C=I.C, d=I.d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)

                flatten_lb = np.ndarray.flatten(lb, 'F')
                map = np.argwhere(flatten_lb < 0)
                lb_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_lb[0]) + map[0][i]
                    index = map[i][1]
                    lb_map = np.append(lb_map, index)

                map = np.argwhere(flatten_ub > 0)
                ub_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_ub[0]) + map[0][i]
                    index = map[i][1]
                    ub_map = np.append(ub_map, index)

                lu_map_float = np.intersect1d([lb_map], [ub_map])
                lu_map = np.array(lu_map_float, dtype=np.int)
                listed_lu_map = lu_map.tolist()
                m = len(listed_lu_map)

                #In = np.array([In])
                In_list = []
                In_list.append(In)
                for i in range(m):
                    if dis_opt == 'display':
                        print('\nPerforming exact PosLin_%d operation using Star', lu_map[i])
                    In_list = PosLin.stepReachMultipleInputs(In_list, listed_lu_map[i], option, lp_solver)
                S = In_list
                return S
        else:
            # S = np.array([])
            S = []
            return S

    # # exact reachability analysis using star
    # def reach_star_exact_multipleInputs(*args):
    #     # @I: star input sets
    #     # @option: = 'parallel' use parallel computing
    #     #          = not declare -> don't use parallel computing
    #     if len(args) == 1: # 2 args
    #         In = args[0]
    #         option = args[1]
    #         dis_opt = ''
    #         lp_solver = 'gurobi'
    #     elif len(args) == 2:  # 3 args
    #         In = args[0]
    #         option = args[1]
    #         dis_opt = args[2]
    #         lp_solver = 'gurobi'
    #     elif len(args) == 3:  # 4 args
    #         In = args[0]
    #         option = args[1]
    #         dis_opt = args[2]
    #         lp_solver = args[3]
    #     else:
    #         'error: Invalid number of input arguments, should be 2, 3, or 4'
    #
    #     n = len(In)
    #     # S = np.array([])
    #     S = []
    #     if len(option) - 1 == 0 or option == 'single':
    #         for i in range (n):
    #             S1 = PosLin.reach_star_exact(In[i], [], dis_opt, lp_solver)
    #             if len(S):
    #                 S.extend(S)
    #                 S.extend(S1)
    #                 del S[:2]
    #             else:
    #                 S.extend(S1)
    #             # S = np.vstack([S, PosLin.reach_star_exact(In[i], [], dis_opt, lp_solver)])
    #     # elif option == 'parallel':
    #     #     for i in prange (n):
    #     #         S = np.vstack([S, PosLin.reach_star_exact(In[i], [], dis_opt, lp_solver)])
    #     else: 'error: unknown computation option'
    #     return S

    # step reach approximation using star
    def stepReachStarApprox(I, index):
        # @I: Star set input
        # @index: index of the neuron performing stepReach
        # @S: star output
        from engine.set.star import Star
        from engine.set.zono import Zono

        assert isinstance(I, Star), 'error: input set is not a star set'

        # using estimateRange function to get lb and ub for now
        [lb, ub] = I.estimateRange(index)
        #lb = I.getMin(index)

        if lb > 0:
            S = I
            return S
        else:
            #ub = I.getMax(index)
            if ub <= 0:
                V = copy.deepcopy(I.V)
                V[index, :] = 0
                if not isinstance(I.Z, Zono):
                    c1 = copy.deepcopy(I.Z.c)
                    c1[index] = 0
                    V1 = copy.deepcopy(I.Z.V)
                    V1[index, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = np.array([])
                S = Star(V=V, C=I.C, d=I.d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)
                return S
            else:
                print('\nAdd a new predicate variables at index = %d', index)
                n = I.nVar + 1

                # over-approximation constraints
                # constraint 1: y[index] = ReLU(x[index]) >= 0
                C1 = np.zeros([n])
                C1[n-1] = -1
                d1 = 0
                # constraint 2: y[index] >= x[index]
                C2 = np.column_stack([I.V[index, 1:n], -1])
                d2 = copy.deepcopy(-I.V[index, 0])
                # constraint 3: y[index] <= ub(x[index] - lb)/(ub - lb)
                C3 = np.column_stack([-(ub/(ub-lb))*I.V[index, 1:n], 1])
                d3 = -ub*lb/(ub-lb)+ub * I.V[index, 0] / (ub-lb)

                m = I.C.shape[0]
                C0 = np.column_stack([I.C, np.zeros([m, 1])])
                d0 = copy.deepcopy(I.d)
                new_C = np.vstack([C0, C1, C2, C3])
                new_d = np.vstack([d0, d1, d2, d3])
                new_V = np.column_stack([I.V, np.zeros([I.dim, 1])])
                new_V[index, :] = np.zeros([1, n+1])
                new_V[index, n] = 1
                new_predicate_lb = np.row_stack([I.predicate_lb, 0])
                new_predicate_ub = np.row_stack([I.predicate_ub, ub])

                # update outer-zono
                lamda = ub/(ub-lb)
                mu = -0.5*ub*lb/(ub-lb)
                if isinstance(I.Z, Zono):
                    c = copy.deepcopy(I.Z.c)
                    c[index] = lamda * c[index] + mu
                    V = copy.deepcopy(I.Z.V)
                    V[index, :] = lamda * V[index, :]
                    I1 = np.zeros([I.dim, 1])
                    I1[index] = mu
                    V = np.column_stack([V, I1])
                    new_Z = Zono(c, V)
                else:
                    new_Z = np.array([])

                S = Star(V=new_V, C=new_C, d=new_d, pred_lb=new_predicate_lb, pred_ub=new_predicate_ub, outer_zono=new_Z)
                return S

    # over-approximate reachability analysis using Star
    def reach_star_approx(I):
        # @I: star input set
        # @S: star output set
        from engine.set.star import Star
        from engine.set.zono import Zono

        assert isinstance(I, Star), 'error: input set is not a star set'

        #if Star.isEmptySet(I):
        if 1 == 0:
            #S = np.array([])
            S = []
            return S
        else:
            [lb, ub] = Star.estimateRanges(I)
            if len(lb) == 0 or len(ub) == 0:
                #S = np.array([])
                S = []
                return S
            else:
                flatten_ub = np.ndarray.flatten(ub, 'F')
                map = np.argwhere(flatten_ub <= 0)
                ub_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_ub[0]) + map[0][i]
                    index = map[i][1]
                    print(index)
                    ub_map = np.append(ub_map, index)
                V = copy.deepcopy(I.V)
                #V[ub_map, :] = 0

                if isinstance(I.Z, Zono):
                    c1 = copy.deepcopy(I.Z.c)
                    #c1[ub_map, :] = 0
                    V1 = copy.deepcopy(I.Z.V)
                    #V1[ub_map, :] = 0
                    new_Z = Zono(c1, V1)
                else:
                    new_Z = np.array([])

                In = Star(V=V, C=I.C, d=I.d, pred_lb=I.predicate_lb, pred_ub=I.predicate_ub, outer_zono=new_Z)

                flatten_lb = np.ndarray.flatten(lb, 'F')
                map = np.argwhere(flatten_lb < 0)
                lb_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_lb[0]) + map[0][i]
                    index = map[i][1]
                    lb_map = np.append(lb_map, index)

                map = np.argwhere(flatten_ub > 0)
                ub_map = np.array([])
                for i in range(len(map)):
                    #index = map[i][1] * len(flatten_ub[0]) + map[0][i]
                    index = map[i][1]
                    ub_map = np.append(ub_map, index)

                lu_map_float = np.intersect1d([lb_map], [ub_map])
                lu_map = np.array(lu_map_float, dtype=np.int)
                listed_lu_map = lu_map.tolist()
                m = len(listed_lu_map)

                for i in range(m):
                    print('\nPerforming approximate PosLin_%d operation using Star', lu_map[i])
                    In = PosLin.stepReachStarApprox(In, listed_lu_map[i])

                S = In
                return S


    # # step reach approximation using star
    # def multipleStepReachStarApprox_at_one(I, index, lb, ub):
    #     # @I: star set input
    #     # @index: index of the neurons performing stepReach
    #     # @lb: lower bound of x[index]
    #     # @ub: upper bound of x[index]
    #
    #     from engine.set.star import Star
    #     assert isinstance(I, Star), 'error: input set is not a star set'
    #
    #     N = I.dim
    #     m = len(index) # number of neurons involved (number of new predicate variables introduced)
    #
    #     # construct new basis array
    #     V1 = copy.deepcopy(I.V) # originaial basis array
    #     V1[index, :] = 0
    #     V2 = np.zeros([N, m]) # basis array for new predicates
    #     for i in range (m):
    #         V2[index[i], i] = 1
    #     new_V = np.column_stack([V1, V2]) # new basis for over-approximate star set
    #
    #     # construct new constraints on new predicate variables
    #     # case 0: keep the old constraints on the old predicate variable
    #     n = I.nVar # number of old predicate variables
    #     C0 = np.column_stack([I.C, np.zeros([I.C.shape[0], m])])
    #     d0 = copy.deepcopy(I.d)
    #
    #     # case 1: y[index] >= 0
    #     C1 = np.column_stack([np.zeros([m, n]), -np.identity([m])])
    #     d1 = np.zeros([m ,1])
    #
    #     # case 2: y[index] >= x[index]
    #     C2 = np.column_stack([I.V[index, 1:n+1], -V2[index, 0:m]])
    #     d2 = copy.deepcopy(-I.V[index, 0])
    #
    #     # case 3: y[index] <= (ub/(ub - lb))*(x-lb)
    #     # add....
    #     a = ub/(ub-lb) # devide element-wise
    #     b = np.multiply(a, lb) # multiply element-wise
    #     C3 = np.column_stack([np.multiply(-a, I.V[index, 1:n+1]), V2[index, 0:m]])
    #     d3 = np.multiply(a, I.V[index, 0]) - b
    #
    #     new_C = np.vstack([C0, C1, C2, C3])
    #     new_d = np.vstack([d0, d1, d2, d3])
    #
    #     new_pred_lb = np.vstack([I.predicate_lb, np.zeros([m, 1])])
    #     new_pred_ub = np.vstack([I.predicate_ub, ub])
    #
    #     S = Star(V=new_V, C=new_C, d=new_d, pred_lb=new_pred_lb, pred_ub=new_pred_ub)
    #     return S

    # # more efficient method by doing multiple stepReach at one time
    # # over-approximate reachability analysis using Star
    # def reach_star_approx2(*args):
    #     # @I: star input set
    #     # @option: 'parallel' or single
    #     # @S: star output set
    #
    #     if len(args) == 1:
    #         I = args[0]
    #         option = 'single'
    #         dis_opt = ''
    #         lp_solver = 'glpk'
    #     elif len(args) == 2:
    #         I = args[0]
    #         option = args[1]
    #         dis_opt = ''
    #         lp_solver = 'glpk'
    #     elif len(args) == 3:
    #         I = args[0]
    #         option = args[1]
    #         dis_opt = args[2]
    #         lp_solver = 'glpk'
    #     elif len(args) == 4:
    #         I = args[0]
    #         option = args[1]
    #         dis_opt = args[2]
    #         lp_solver = args[3]
    #     else:
    #         'error: Invalid number of input arguments, should be 1, 2, 3, or 4'
    #
    #     from engine.set.star import Star
    #     assert isinstance(I, Star), 'error: input set is not a star set'
    #
    #     if Star.isEmptySet(I):
    #         S = np.array([])
    #         return S
    #     else:
    #         [lb, ub] = I.estimateRanges;
    #         if len(lb) == 0 or len(ub) == 0:
    #             S = np.array([])
    #             return S
    #         else:
    #             # find all indexes having ub <= 0, then reset the
    #             # values of the elements corresponding to these indexes to 0
    #             flatten_ub = np.ndarray.flatten(ub, 'F')
    #             flatten_lb = np.ndarray.flatten(lb, 'F')
    #
    #             if dis_opt == 'display':
    #                 print('\nFinding all neurons (in %d neurons) with ub <= 0...', len(ub))
    #
    #             map = np.argwhere(flatten_ub <= 0)
    #             map1 = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_ub[0]) + map[0][i]
    #                 index = map[i][1]
    #                 map1 = np.append(map1, index)
    #
    #             if dis_opt == 'display':
    #                 print('\n%d neurons with ub <= 0 are found by estimating ranges', len(map1))
    #
    #             map = np.argwhere(flatten_lb < 0)
    #             lb_map = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_lb[0]) + map[0][i]
    #                 index = map[i][1]
    #                 lb_map = np.append(lb_map, index)
    #
    #             map = np.argwhere(flatten_ub > 0)
    #             ub_map = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_ub[0]) + map[0][i]
    #                 index = map[i][1]
    #                 ub_map = np.append(ub_map, index)
    #
    #             map2 = np.intersect1d([lb_map], [ub_map])
    #
    #             if dis_opt == 'display':
    #                 print('\nFinding neurons (in %d neurons) with ub <= 0 by optimizing ranges: ', len(map2))
    #
    #             xmax = I.getMaxs(map2, option, dis_opt, lp_solver)
    #             flatten_xmax = np.ndarray.flatten(xmax, 'F')
    #             map = np.argwhere(flatten_xmax <= 0)
    #             map3 = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_xmax[0]) + map[0][i]
    #                 index = map[i][1]
    #                 map3 = np.append(map3, index)
    #
    #             if dis_opt == 'display':
    #                 print('\n%d neurons (in %d neurons) with ub <= 0 are found by optimizing ranges', len(map3), len(map2))
    #
    #             n = len(map3)
    #             map4 = np.zeros(n, 1)
    #             for i in range (n):
    #                 map4[i] = map2[map3[i]]
    #
    #             map11 = np.vstack([map1, map4])
    #             In = I.resetRow(map11) # reset to zero at the element having ub <= 0, need to add resetRow func in star
    #             if dis_opt == 'display':
    #                 print('\n(%d+%d =%d)/%d neurons have ub <= 0', len(map1), len(map3), len(map11), len(ub))
    #
    #             # find all indexes that have lb < 0 & ub > 0, then
    #             # apply the over-approximation rule for ReLU
    #
    #             if dis_opt == 'display':
    #                 print("\nFinding all neurons (in %d neurons) with lb < 0 & ub >0: ", len(ub))
    #
    #             map = np.argwhere(flatten_xmax > 0)
    #             map5 = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_xmax[0]) + map[0][i]
    #                 index = map[i][1]
    #                 map5 = np.append(map5, index)
    #
    #             map6 = map2[map5[:]] # all indexes having ub > 0
    #             xmax1 = xmax[map5[:]] # upper bound of all neurons having ub > 0
    #
    #             xmin = I.getMins(map6, option, dis_opt, lp_solver)
    #             flatten_xmin = np.ndarray.flatten(xmin, 'F')
    #             map = np.argwhere(flatten_xmin < 0)
    #             map7 = np.array([])
    #             for i in range(len(map)):
    ##                 index = map[i][1] * len(flatten_xmin[0]) + map[0][i]
    #                 index = map[i][1]
    #                 map7 = np.append(map7, index)
    #
    #             map8 = map6[map7[:]] # all indexes habing lb < 0 & ub > 0
    #             lb1 = xmin[map7[:]] # lower bound of all indexes having lb < 0 & ub > 0
    #             ub1 = xmax1[map7[:]] # upper bound of all neurons having lb < 0 & ub > 0
    #
    #             if dis_opt == 'display':
    #                 print('\n%d/%d neurons have lb < 0 & ub > 0', len(map8), len(ub))
    #                 print('\nConstruct new star set, %d new predicate variables are introduced', len(map8))
    #
    #             S = PosLin.multipleStepReachStarApprox_at_one(In, map8, lb1, ub1) # one-shot approximation
    #             return S


    # main function for reachability analysis
    def reach(*args):

        if len(args) == 6: # 6 arguments
            I = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3] # used for aprx-star only
            dis_opt = args[4] # display option
            lp_solver = args[5]
        elif len(args) == 5: # 5 arguments
            I = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3] # used for aprx-star only
            dis_opt = args[4] # display option
            lp_solver = 'gurobi'
        elif len(args) == 4: # 4 arguments
            I = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = args[3] # used for aprx-star only
            dis_opt = '' # display option
            lp_solver = args[5]
        elif len(args) == 3: # 3 arguments
            I = args[0]
            method = args[1]
            option = args[2]
            relaxFactor = 0 # used for aprx-star only
            dis_opt = '' # display option
            lp_solver = 'gurobi'
        elif len(args) == 2: # 2 arguments
            I = args[0]
            method = args[1]
            option = 'parallel'
            relaxFactor = 0 # used for aprx-star only
            dis_opt = '' # display option
            lp_solver = 'gurobi'
        elif len(args) == 1: # 1 arguments
            I = args[0]
            method = 'exact-star'
            option = 'parallel'
            relaxFactor = 0 # used for aprx-star only
            dis_opt = '' # display option
            lp_solver = 'gurobi'
        else: 'error: Invalid number of input arguments (should be 1, 2, 3, 4, or 5)'

        if method == 'exact-star': # exact analysis using star
            # R = PosLin.reach_star_exact_multipleInputs(I, option, dis_opt, lp_solver)
            R = PosLin.reach_star_exact(I, np.matrix([]))
            return R
        # elif method == 'exact-polyhedron': # exact analysis using polyhedron
        #     R = PosLin.reach_polyhedron_exact(I, option, dis_opt)
        elif method == 'approx-star': # over-approximate analysis using star
            #R = PosLin.reach_star_approx(I, option, dis_opt, lp_solver)
            R = PosLin.reach_star_approx(I)
            return R
