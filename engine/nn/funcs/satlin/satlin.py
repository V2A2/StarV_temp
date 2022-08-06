#!/usr/bin/python3
from re import A
import sys
import copy
from traceback import print_last
import numpy as np

sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/set/zono/")
sys.path.insert(0, "engine/set/box/")

# ------------- NOTEs: preprocessing library which contains the normalize function to normalize -------------
# ------------- the data. It takes an array in as an input and normalizes its values between 00 -------------
# ------------- and 11. It then returns an output array with the same dimensions as the input. -------------

# from sklearn import preprocessing
# from numba import njit, prange


class SatLin:
    # ------------- SATLIN : class for computing reachable set of Satlin Transfer Function -------------
    # ------------- Reference: https://www.mathworks.com/help/deeplearning/ref/satlin.html -------------

    # ------------- evaluate method and reachability analysis with stars  -------------

    def evaluate(x):
        """
        evaluate method and reachability analysis with stars

        Args:
            @x = np.arrays

        Returns:
            0, if n <= 0
            n, if 0 <= n <= 1
            1, if 1 <= n
        """

        a = np.maximum(x, 0)
        b = np.minimum(a, 1)
        # print("\n b ------------------------ \n", b)
        return b

    def stepReach(*args):
        """
        stepReach method, compute reachable set for a single step

        Args:
            @I: single star set input
            @index: index of current x[index] of current step (should be the number from matlab - 1)

        Others:
            @xmin: min of x[index]
            @xmax: max of x[index]

        Returns:
            @S: star output set
        """

        len_args = len(args)
        if len_args == 2:  # 2 arguments
            [I, index] = args
            lp_solver = 'gurobi'
        elif len_args == 3:  # 3 arguments
            [I, index, lp_solver] = args
        else:
            'error: Invalid number of input arguments, should be 2 or 3'

        from star import Star
        from zono import Zono

        assert isinstance(I, Star), 'error: input set is not a star set'

        # ------------- TODO: Using getMin and getMax after renew the gurobi license -------------
        xmin = I.getMin(index, lp_solver)
        xmax = I.getMax(index, lp_solver)
        # [xmin, xmax] = I.estimateRange(index)
        # print("\n xmax ------------------------ \n", xmax)
        # print("\n xmin ------------------------ \n", xmin)
        C = copy.deepcopy(I.C)
        # print("\n C ------------------------ \n", C)
        d = copy.deepcopy(I.d)
        # print("\n d ------------------------ \n", d)
        c1 = copy.deepcopy(I.V[index, 0])
        # print("\n c1 ------------------------ \n", c1)
        V1 = copy.deepcopy(I.V[index, 1:I.nVar + 1])
        # print("\n V1 ------------------------ \n", V1)

        # ------------- case 1) only single set -------------
        if xmin >= 0 and xmax <= 1:
            S = []
            S.append(I)
            return S

        # ------------- case 2) -------------
        if xmin >= 0 and xmax > 1:
            # ------------- x >= 0 and x <= 1 -------------
            new_C1 = np.vstack([C, -V1, V1])
            # print("\n new_C1 ------------------------ \n", new_C1)
            new_d1 = np.hstack([d, c1, 1 - c1])
            # print("\n new_d1 ------------------------ \n", new_d1)
            S1 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub, I.Z)
            # print("\n S1 ------------------------ \n", S1)

            # ------------- x > 1 -------------
            new_V2 = copy.deepcopy(I.V)
            # print("\n new_V2 ------------------------ \n", new_V2)
            new_V2[index, :] = 0
            new_V2[index, 0] = 1
            # print("\n new_V2 ------------------------ \n", new_V2)
            if isinstance(I.Z, Zono):
                new_Z2 = copy.deepcopy(I.Z)
                # print("\n new_Z2 ------------------------ \n", new_Z2)
                new_Z2.c[index] = 1
                # print("\n new_Z2.c ------------------------ \n", new_Z2.c)
                new_Z2.V[index, :] = 0
                # print("\n new_Z2.V ------------------------ \n", new_Z2.V)
            else:
                new_Z2 = np.array([])

            new_C2 = np.vstack([C, -V1])
            # print("\n new_C2 ------------------------ \n", new_C2)
            new_d2 = np.hstack([d, -1 + c1])
            # print("\n new_d2 ------------------------ \n", new_d2)
            S2 = Star(new_V2, new_C2, new_d2, I.predicate_lb, I.predicate_ub,
                      new_Z2)
            # print("\n S2 ------------------------ \n", S2)
            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 3) -------------
        if xmin < 0 and xmax > 0 and xmax <= 1:
            # ------------- 1 >= x >= 0 -------------
            new_C1 = np.vstack([C, -V1])
            # print("\n new_C1 ------------------------ \n", new_C1)
            new_d1 = np.hstack([d, c1])
            # print("\n new_d1 ------------------------ \n", new_d1)
            S1 = Star(I.V, new_C1, new_d1, I.predicate_lb, I.predicate_ub, I.Z)
            # print("\n S1 ------------------------ \n", S1)

            # ------------- x < 0 -------------
            new_V2 = copy.deepcopy(I.V)
            # print("\n new_V2 ------------------------ \n", new_V2)
            new_V2[index, :] = 0
            # print("\n new_V2 ------------------------ \n", new_V2)
            if isinstance(I.Z, Zono):
                new_Z2 = copy.deepcopy(I.Z)
                # print("\n new_Z2 ------------------------ \n", new_Z2)
                new_Z2.c[index] = 0
                # print("\n new_Z2.c ------------------------ \n", new_Z2.c)
                new_Z2.V[index, :] = 0
                # print("\n new_Z2.V ------------------------ \n", new_Z2.V)
            else:
                new_Z2 = np.array([])

            new_C2 = np.vstack([C, V1])
            # print("\n new_C2 ------------------------ \n", new_C2)
            new_d2 = np.hstack([d, -c1])
            # print("\n new_d2 ------------------------ \n", new_d2)
            S2 = Star(new_V2, new_C2, new_d2, I.predicate_lb, I.predicate_ub,
                      new_Z2)
            # print("\n S2 ------------------------ \n", S2)

            S = []
            S.append(S1)
            S.append(S2)
            return S

        # ------------- case 4) -------------
        if xmin < 0 and xmax > 1:
            # ------------- x < 0 -------------
            new_C1 = np.vstack([C, V1])
            # print("\n new_C1 ------------------------ \n", new_C1)
            new_d1 = np.hstack([d, -c1])
            # print("\n new_d1 ------------------------ \n", new_d1)
            new_V1 = copy.deepcopy(I.V)
            # print("\n new_V1 ------------------------ \n", new_V1)
            new_V1[index, :] = 0
            # print("\n new_V1 ------------------------ \n", new_V1)
            if isinstance(I.Z, Zono):
                new_Z1 = copy.deepcopy(I.Z)
                # print("\n new_Z1 ------------------------ \n", new_Z1)
                new_Z1.c[index] = 0
                # print("\n new_Z1.c ------------------------ \n", new_Z1.c)
                new_Z1.V[index, :] = 0
                # print("\n new_Z1.V ------------------------ \n", new_Z1.V)
            else:
                new_Z1 = np.array([])

            S1 = Star(new_V1, new_C1, new_d1, I.predicate_lb, I.predicate_ub,
                      new_Z1)
            # print("\n S1 ------------------------ \n", S1)

            # ------------- 0 <= x <= 1 -------------
            new_C2 = np.vstack([C, -V1, V1])
            # print("\n new_C2 ------------------------ \n", new_C2)
            new_d2 = np.hstack([d, c1, 1 - c1])
            # print("\n new_d2 ------------------------ \n", new_d2)
            S2 = Star(I.V, new_C2, new_d2, I.predicate_lb, I.predicate_ub, I.Z)
            # print("\n S2 ------------------------ \n", S2)

            # ------------- x > 1 -------------
            new_C3 = np.vstack([C, -V1])
            # print("\n new_C3 ------------------------ \n", new_C3)
            new_d3 = np.hstack([d, -1 + c1])
            # print("\n new_d3 ------------------------ \n", new_d3)
            new_V3 = copy.deepcopy(I.V)
            # print("\n new_V3 ------------------------ \n", new_V3)
            new_V3[index, :] = 0
            new_V3[index, 0] = 1
            # print("\n new_V3 ------------------------ \n", new_V3)
            if isinstance(I.Z, Zono):
                new_Z3 = copy.deepcopy(I.Z)
                # print("\n new_Z3 ------------------------ \n", new_Z3)
                new_Z3.c[index] = 1
                # print("\n new_Z3.c ------------------------ \n", new_Z3.c)
                new_Z3.V[index, :] = 0
                # print("\n new_Z3.V ------------------------ \n", new_Z3.V)
            else:
                new_Z3 = np.array([])
            S3 = Star(new_V3, new_C3, new_d3, I.predicate_lb, I.predicate_ub,
                      new_Z3)
            # print("\n S3 ------------------------ \n", S3)

            S = []
            S.append(S1)
            S.append(S2)
            S.append(S3)
            return S

        # ------------- case 5) -------------
        if xmin >= 1:
            new_V = copy.deepcopy(I.V)
            # print("\n new_V ------------------------ \n", new_V)
            new_V[index, :] = 0
            new_V[index, 0] = 1
            # print("\n new_V ------------------------ \n", new_V)
            if isinstance(I.Z, Zono):
                new_Z = copy.deepcopy(I.Z)
                # print("\n new_Z ------------------------ \n", new_Z)
                new_Z.c[index] = 1
                # print("\n new_Z.c ------------------------ \n", new_Z.c)
                new_Z.V[index, :] = 0
                # print("\n new_Z.V ------------------------ \n", new_Z.V)
            else:
                new_Z = np.array([])
            S = []
            S1 = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
            S.append(S1)
            # print("\n S ------------------------ \n", S)
            return S

        # ------------- case 6) -------------
        if xmax <= 0:
            new_V = copy.deepcopy(I.V)
            # print("\n new_V ------------------------ \n", new_V)
            new_V[index, :] = 0
            # print("\n new_V ------------------------ \n", new_V)
            if isinstance(I.Z, Zono):
                new_Z = copy.deepcopy(I.Z)
                # print("\n new_Z ------------------------ \n", new_Z)
                new_Z.c[index] = 0
                # print("\n new_Z.c ------------------------ \n", new_Z.c)
                new_Z.V[index, :] = 0
                # print("\n new_Z.V ------------------------ \n", new_Z.V)
            else:
                new_Z = np.array([])
            S = []
            S1 = Star(new_V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
            S.append(S1)
            # print("\n S ------------------------ \n", S)
            return S

    def stepReachMultipleInputs(*args):
        """
        stepReach with multiple inputs

        Args:
            @I: an list of stars
            @index: index where stepReach is performed
            @option: = 'parallel' use parallel computing
                     = not declare -> don't use parallel computing

        Returns:
            @S: star output set
        """

        len_args = len(args)
        if len_args == 3:  # 3 arguments
            [I, index, option] = args
            lp_solver = 'gurobi'
        elif len_args == 4:  # 4 arguments
            [I, index, option, lp_solver] = args
        else:
            'error: Invalid number of input arguments, should be 3 or 4'

        from star import Star

        assert isinstance(I, list), 'error: input is not an array star sets'
        # print("\n I_list ------------------------ \n", I)
        assert isinstance(I[0],
                          Star), 'error: input at index 0 is not a star set'
        # print("\n I[0] ------------------------ \n", I[0])

        p = len(I)
        # print("\n p ------------------------ \n", p)
        S = []

        if len(option) == 0:
            for i in range(p):
                # print("\n i ------------------------ \n", i)
                S1 = SatLin.stepReach(I[i], index, lp_solver)
                # print("\n S1 ------------------------ \n", len(S1))
                # print("\n S1 ------------------------ \n", (S1))
                S.extend(S1)
                # print("\n S ------------------------ \n", len(S))
            return S

        # ------------- TODO: Fix parallel part -------------
        # elif option == 'parellel':
        #     #@njit(parallel=True)
        #     for i in prange(p):
        #         S = np.column_stack([S, PosLin.stepReach(I[i], index, lp_solver)])
        #     return S
        else:
            'error: Unknown option'

    def reach_star_exact(*args):
        """
        exact reachability analysis using star

        Args:
            @I: a list of star input sets
            @option: = 'parallel' use parallel computing
                     = not declare -> don't use parallel computing
        Returns:
            @S: star output set
        """

        len_args = len(args)
        if len_args == 2:  # 2 arguments
            [I, option] = args
            dis_opt = ""
            lp_solver = 'gurobi'
        elif len_args == 3:  # 3 arguments
            [I, option, dis_opt] = args
            lp_solver = 'gurobi'
        elif len_args == 4:  # 4 arguments
            [I, option, dis_opt, lp_solver] = args
        else:
            'error: Invalid number of input arguments, should be 2, 3 or 4'

        from star import Star
        from zono import Zono

        # ------------- TODO: Fix gurobi license -------------
        # if not I.isEmptySet():
        if isinstance(I, Star):
            [lb, ub] = I.estimateRanges()
            # print("\n lb ------------------------ \n", lb)
            # print("\n ub ------------------------ \n", ub)
            map1 = np.argwhere(ub <= 0)  # computation map
            # print("\n map1 ------------------------ \n", map1)
            V = copy.deepcopy(I.V)
            # print("\n V ------------------------ \n", V)
            V[map1, :] = 0
            # print("\n V ------------------------ \n", V)
            # ------------- update outer zono -------------
            map2 = np.argwhere(lb >= 1)
            # print("\n map2 ------------------------ \n", map2)

            # ------------- TODO: should this be map2? -------------
            V[map1, :] = 0
            V[map1, 0] = 1
            # print("\n V ------------------------ \n", V)
            if isinstance(I.Z, Zono):
                c1 = copy.deepcopy(I.Z.c)
                # print("\n c1 ------------------------ \n", c1)
                c1[map1] = 0
                V1 = copy.deepcopy(I.Z.V)
                # print("\n V1 ------------------------ \n", V1)
                V1[map1, :] = 0
                c1[map2] = 1
                V1[map2, :] = 0
                # print("\n V1 ------------------------ \n", V1)
                # print("\n c1 ------------------------ \n", c1)

                new_Z = Zono(c1, V1)
                # print("\n new_Z ------------------------ \n", new_Z)
            else:
                new_Z = np.array([])

            In = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub, new_Z)
            # print("\n In ------------------------ \n", In)

            lb_map = np.argwhere(lb < 1)
            # print("\n lb_map ------------------------n", lb_map)
            ub_map = np.argwhere(ub > 0)
            # print("\n ub_map ------------------------n", ub_map)
            map_float = np.intersect1d([lb_map], [ub_map])
            map = np.array(map_float, dtype=np.int)
            # map_to_list = map.tolist()
            # print("\n map_to_list ------------------------ \n", map_to_list)
            m = len(map)
            # print("\n m ------------------------n", m)

            In_list = []
            In_list.append(In)
            for i in range(m):
                # print("\n i ------------------------ \n", i)
                if dis_opt == 'display':
                    print("\n Performing exact PosLin_%d operation using Star",
                          map[i])
                # print("\n In_list ------------------------ before \n", In_list)
                In_list = SatLin.stepReachMultipleInputs(
                    In_list, map[i], option, lp_solver)
                # print("\n In_list ------------------------ after \n", In_list)
            S = In_list
            return S
        else:
            S = []
            return S

    def stepReachStarApprox(I, index, lp_solver):
        """
        step over-approximate reachability analysis using Star

        Args:
            @I: Star set input
            @index: index of the neuron performing stepReach

        Returns:
            @S: star output set
        """

        from star import Star
        assert isinstance(I, Star), 'error: input set is not a star set'

        # ------------- TODO: Call get Mins Maxs -------------
        # print("\n lp_solver ------------------------ \n", lp_solver)
        lb = I.getMin(index, lp_solver)
        ub = I.getMax(index, lp_solver)
        # [lb, ub] = I.estimateRange(index)
        # print("\n lb ------------------------ \n", lb)
        # print("\n ub ------------------------ \n", ub)

        if ub <= 0:
            V = copy.deepcopy(I.V)
            # print("\n V ------------------------ \n", V)
            V[index, :] = np.zeros([1, I.nVar + 1])
            # print("\n V ------------------------ \n", V)
            S = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub)
            return S

        if lb >= 1:
            V = copy.deepcopy(I.V)
            # print("\n V ------------------------ \n", V)
            V[index, :] = np.zeros([1, I.nVar + 1])
            V[index, 0] = 1
            # print("\n V ------------------------ \n", V)
            S = Star(V, I.C, I.d, I.predicate_lb, I.predicate_ub)
            return S

        if 1 > lb and lb > 0 and ub > 1:
            # ------------- constraint 1: y[index] <= x[index] -------------
            C1 = np.hstack([-I.V[index, 1:I.nVar + 1], 1])
            # print("\n C1 ------------------------ \n", C1)
            d1 = copy.deepcopy(I.V[index, 0])
            # print("\n d1 ------------------------ \n", d1)
            # ------------- constraint 2: y[index] <= 1 -------------
            C2 = np.zeros([1, I.nVar + 1])
            # print("\n C2 ------------------------ \n", C2)
            C2[0, I.nVar] = 1
            # print("\n C2 ------------------------ \n", C2)
            d2 = 1
            # ------------- constraint 3: y[index] >= ((1-lb)/(ub-lb))(x-lb) + lb -------------
            a = (1 - lb) / (ub - lb)
            # print("\n a ------------------------ \n", a)
            C3 = np.hstack([a * I.V[index, 1:I.nVar + 1], -1])
            # print("\n C3 ------------------------ \n", C3)
            d3 = -lb + a * lb - a * I.V[index, 0]
            # print("\n d3 ------------------------ \n", d3)

            m = I.C.shape[0]
            # print("\n m ------------------------ \n", m)
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            # print("\n C0 ------------------------ \n", C0)
            d0 = copy.deepcopy(I.d)
            # print("\n d0 ------------------------ \n", d0)
            new_C = np.vstack([C0, C1, C2, C3])
            # print("\n new_C ------------------------ \n", new_C)
            new_d = np.hstack([d0, d1, d2, d3])
            # print("\n new_d ------------------------ \n", new_d)
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            # print("\n new_V ------------------------ \n", new_V)
            new_V[index, :] = np.zeros([1, I.nVar + 2])
            new_V[index, I.nVar + 1] = 1
            # print("\n new_V ------------------------ \n", new_V)
            new_lb = np.hstack([I.predicate_lb, lb])
            # print("\n predicate_lb ------------------------ \n", I.predicate_lb)
            # print("\n new_lb ------------------------ \n", new_lb)
            new_ub = np.hstack([I.predicate_ub, 1])
            # print("\n predicate_ub ------------------------ \n", I.predicate_ub)
            # print("\n new_ub ------------------------ \n", new_ub)

            S = Star(new_V, new_C, new_d, new_lb, new_ub)
            return S

        if lb >= 0 and ub <= 1:
            S = Star(I.V, I.C, I.d, I.predicate_lb, I.predicate_ub)
            return S

        if lb < 0 and 0 < ub and ub <= 1:
            n = I.nVar + 1
            # print("\n n ------------------------ \n", n)
            # ------------- over-approximation constraints -------------
            # ------------- constraint 1: y[index] >= 0 -------------
            C1 = np.zeros([1, n])
            # print("\n C1 ------------------------ \n", C1)
            C1[0, n - 1] = -1
            # print("\n C1 ------------------------ \n", C1)
            d1 = 0
            # ------------- constraint 2: y[index] >= x[index] -------------
            C2 = np.hstack([I.V[index, 1:n], -1])
            # print("\n C2 ------------------------ \n", C2)
            d2 = copy.deepcopy(-I.V[index, 0])
            # print("\n d2 ------------------------ \n", d2)
            # ------------- constraint 3: y[index] <= ub(x[index] -lb)/(ub - lb) -------------
            a = ub / (ub - lb)
            # print("\n a ------------------------ \n", a)
            C3 = np.hstack([-a * I.V[index, 1:n], 1])
            # print("\n C3 ------------------------ \n", C3)
            d3 = a * I.V[index, 0] - a * lb
            # print("\n d3 ------------------------ \n", d3)

            m = I.C.shape[0]
            # print("\n m ---------------------------------- \n", m)
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            # print("\n C0 ------------------------ \n", C0)
            d0 = copy.deepcopy(I.d)
            # print("\n d0 ------------------------ \n", d0)
            new_C = np.vstack([C0, C1, C2, C3])
            # print("\n new_C ------------------------ \n", new_C)
            new_d = np.hstack([d0, d1, d2, d3])
            # print("\n new_d ------------------------ \n", new_d)
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            # print("\n new_V ------------------------ \n", new_V)
            new_V[index, :] = np.zeros([1, n + 1])
            new_V[index, n] = 1
            # print("\n new_V ------------------------ \n", new_V)
            new_lb = np.hstack([I.predicate_lb, 0])
            # print("\n predicate_lb ------------------------ \n", I.predicate_lb)
            # print("\n new_lb ------------------------ \n", new_lb)
            new_ub = np.hstack([I.predicate_ub, ub])
            # print("\n predicate_ub ------------------------ \n", I.predicate_ub)
            # print("\n new_ub ------------------------ \n", new_ub)

            S = Star(new_V, new_C, new_d, new_lb, new_ub)
            return S
        if lb < 0 and ub > 1:
            n = I.nVar + 1
            # print("\n n ------------------------ \n", n)
            # ------------- over-approximation constraints -------------
            # ------------- constraint 1: y[index] >= 0 -------------
            C1 = np.zeros([1, n])
            # print("\n C1 ------------------------ \n", C1)
            C1[0, n - 1] = -1
            # print("\n C1 ------------------------ \n", C1)
            d1 = 0

            # ------------- constraint 2: y[index] <= 1 -------------
            C2 = np.zeros([1, n])
            # print("\n C2 ------------------------ \n", C2)
            C2[0, n - 1] = 1
            # print("\n C2 ------------------------ \n", C2)
            d2 = 1

            # ------------- constraint 3: y[index] <= x/(1 -lb) - lb/(1-lb) -------------
            C3 = np.hstack([(-1 / (1 - lb)) * I.V[index, 1:n], 1])
            # print("\n C3 ------------------------ \n", C3)
            d3 = (1 / (1 - lb)) * I.V[index, 0] - lb / (1 - lb)
            # print("\n d3 ------------------------ \n", d3)

            # ------------- constraint 4: y[index] >=  x/ub -------------
            C4 = np.hstack([(1 / ub) * I.V[index, 1:n], -1])
            # print("\n C4 ------------------------ \n", C4)
            d4 = -(1 / ub) * I.V[index, 0]
            # print("\n d4 ------------------------ \n", d4)

            m = I.C.shape[0]
            # print("\n m ------------------------ \n", m)
            C0 = np.hstack([I.C, np.zeros([m, 1])])
            # print("\n C0 ------------------------ \n", C0)
            d0 = copy.deepcopy(I.d)
            # print("\n d0 ------------------------ \n", d0)
            new_C = np.vstack([C0, C1, C2, C3, C4])
            # print("\n new_C ------------------------ \n", new_C)
            new_d = np.hstack([d0, d1, d2, d3, d4])
            # print("\n new_d ------------------------ \n", new_d)
            new_V = np.hstack([I.V, np.zeros([I.dim, 1])])
            # print("\n new_V ------------------------ \n", new_V)
            new_V[index, :] = np.zeros([1, n + 1])
            new_V[index, n] = 1
            # print("\n new_V ------------------------ \n", new_V)
            new_lb = np.hstack([I.predicate_lb, 0])
            # print("\n predicate_lb ------------------------ \n", I.predicate_lb)
            # print("\n new_lb ------------------------ \n", new_lb)
            new_ub = np.hstack([I.predicate_ub, 1])
            # print("\n predicate_ub ------------------------ \n", I.predicate_ub)
            # print("\n new_ub ------------------------ \n", new_ub)

            S = Star(new_V, new_C, new_d, new_lb, new_ub)
            return S

    def reach_star_approx(*args):
        """
        over-approximate reachability analysis using Star

        Args:
            @I: star input set

        Returns:
            @S: star output set
        """

        len_args = len(args)
        if len_args == 1:  # 1 arguments
            I = args[0]
            dis_opt = ''
            lp_solver = 'gurobi'
        elif len_args == 2:  # 2 arguments
            [I, dis_opt] = args
            lp_solver = 'gurobi'
        elif len_args == 3:  # 3 arguments
            [I, dis_opt, lp_solver] = args
        else:
            'error: Invalid number of input arguments, should be 1, 2 or 3'

        from star import Star
        assert isinstance(I, Star), 'error: input set is not a star set'

        # ------------- TODO: Fix gurobi license -------------
        if I.isEmptySet():
        # if I.V.shape[0] == 0:
            S = []
            return S
        else:
            In = I
            for i in range(I.dim):
                if dis_opt == 'display':
                    print(
                        "\n Performing approximate PosLin_%d operation using Star"
                        % i)
                # print("\n lp_solver ------------------------ \n", lp_solver)
                In = SatLin.stepReachStarApprox(In, i, lp_solver)
                # print("\n In ------------------------ \n", In.__repr__())
            S = [In]
            return S

    # ------------- over-approximate reachability analysis use zonotope -------------
    def stepReachZonoApprox(I, index):
        """
        step over-approximate reachability analysis using zonotope
        reference: Fast and Effective Robustness Ceritification,
        Gagandeep Singh, NIPS 2018

        Args:
            @I: zonotope input set
            @index: index of the neuron we want to perform stepReach

        Returns:
            @Z: zonotope output set
        """

        from zono import Zono
        assert isinstance(I, Zono), 'error: input set is not a Zonotope'

        [lb, ub] = I.getRange(index)
        # print("\n lb ------------------------ \n", lb)
        # print("\n ub ------------------------ \n", ub)

        if ub <= 0:
            V = copy.deepcopy(I.V)
            # print("\n V ------------------------ \n", V)
            V[index, :] = np.zeros([1, V.shape[1]])
            # print("\n V ------------------------ \n", V)
            c = copy.deepcopy(I.c)
            # print("\n c ------------------------ \n", c)
            c[index] = 0
            # print("\n c ------------------------ \n", c)
            Z = Zono(c, V)
            return Z
        if lb >= 1:
            V = copy.deepcopy(I.V)
            # print("\n V ------------------------ \n", V)
            V[index, :] = np.zeros([1, V.shape[1]])
            # print("\n V ------------------------ \n", V)
            c = copy.deepcopy(I.c)
            # print("\n c ------------------------ \n", c)
            c[index] = 1
            # print("\n c ------------------------ \n", c)
            Z = Zono(c, V)
            return Z
        if lb >= 0 and ub <= 1:
            Z = Zono(I.c, I.V)
            return Z
        if 1 > lb and lb > 0 and ub > 1:
            c = copy.deepcopy(I.c)
            # print("\n c ------------------------ \n", c)
            c[index] = c[index] + (1 - ub) / 2
            # print("\n c ------------------------ \n", c)
            V = np.zeros([I.dim, 1])
            # print("\n V ------------------------ \n", V)
            V[index] = (1 - ub) / 2
            # print("\n V ------------------------ \n", V)
            V = np.hstack([I.V, V])
            # print("\n V ------------------------ \n", V)
            Z = Zono(c, V)
            return Z
        if lb < 0 and 0 < ub and ub <= 1:
            c = copy.deepcopy(I.c)
            # print("\n c ------------------------ \n", c)
            lamda_opt = ub / (ub - lb)
            # print("\n lamda_opt ------------------------ \n", lamda_opt)
            mu = -0.5 * ub * lb / (ub - lb)
            # print("\n mu ------------------------ \n", mu)
            c[index] = lamda_opt * c[index] + mu
            # print("\n c ------------------------ \n", c)
            V = np.zeros([I.dim, 1])
            # print("\n V ------------------------ \n", V)
            V[index] = mu
            # print("\n V ------------------------ \n", V)
            V = np.hstack([I.V, V])
            # print("\n V ------------------------ \n", V)
            Z = Zono(c, V)
            return Z
        if lb < 0 and ub > 1:
            # ------------- x + 1 -ub <= y <= x - lb, lb <= x <= ub -------------
            c = copy.deepcopy(I.c)
            # print("\n c ------------------------ \n", c)
            mu = (1 + lb - ub) / 2
            # print("\n mu ------------------------ \n", mu)
            c[index] = c[index] - lb + mu
            # print("\n c ------------------------ \n", c)
            V = np.zeros([I.dim, 1])
            # print("\n V ------------------------ \n", V)
            V[index] = mu
            # print("\n V ------------------------ \n", V)
            V = np.hstack([I.V, V])
            # print("\n V ------------------------ \n", V)
            Z = Zono(c, V)
            return Z

    def reach_zono_approx(*args):
        """
        Over-approximate reachability analysis use zonotope
        reference: Fast and Effective Robustness Ceritification,
        Gagandeep Singh, NIPS 2018

        Args:
            @I: zonotope input

        Returns:
            @Z: zonotope output
        """

        from zono import Zono

        if len(args) == 1:
            I = args[0]
            dis_opt = ""
        elif len(args) == 2:
            [I, dis_opt] = args
        else:
            'error: Invalid number of input arguments, should be 1 or 2'

        assert isinstance(I, Zono), 'error: input set is not a Zonotope'
        In = I
        # print("\n In ------------------------ \n", In.__repr__())
        for i in range(I.dim):
            # print("\n i ------------------------ \n", i)
            if dis_opt == 'display':
                print(
                    "\n Performing approximate PosLin_%d operation using Zonotope"
                    % i)
            In = SatLin.stepReachZonoApprox(In, i)
            # print("\n In ------------------------ \n", In.__repr__())

        Z = [In]
        return Z

    # ------------- main reach method -------------
    def reach(*args):
        """
        main function for reachability analysis

        Args:
            @I: a list of star input sets
            @method: 'exact-star' or 'approx-star' or 'approx-zono'
            @option: = 'parallel' use parallel option
                     = '' do use parallel option

        Returns:
            reachability analysis result
        """

        if len(args) == 5:  # 5 arguments
            [I, method, option, dis_opt, lp_solver] = args
        elif len(args) == 4:  # 4 arguments
            [I, method, option, dis_opt] = args
            lp_solver = 'gurobi'
        elif len(args) == 3:  # 3 arguments
            [I, method, option] = args
            dis_opt = ''  # display option
            lp_solver = 'gurobi'
        elif len(args) == 2:  # 2 arguments
            [I, method] = args
            option = 'parallel'
            dis_opt = ''  # display option
            lp_solver = 'gurobi'
        elif len(args) == 1:  # 1 arguments
            I = args[0]
            method = 'exact-star'
            option = 'parallel'
            dis_opt = ''  # display option
            lp_solver = 'gurobi'
        else:
            'error: Invalid number of input arguments (should be 1, 2, 3, or 4)'

        # ------------- TODO: Fix parallel -------------
        option = ''

        if method == 'exact-star':  # exact analysis using star
            R = SatLin.reach_star_exact(I, option, dis_opt, lp_solver)
            return R
        # elif method == 'exact-polyhedron': # exact analysis using polyhedron
        #     R = PosLin.reach_polyhedron_exact(I, option, dis_opt)
        elif method == 'approx-star':  # over-approximate analysis using star
            # R = PosLin.reach_star_approx(I, option, dis_opt, lp_solver)
            # print("\n I ------------------------ \n ", I)
            R = SatLin.reach_star_approx(I, dis_opt, lp_solver)
            return R
        elif method == 'approx-zono':  # over-approximate analysis using zonotope
            R = SatLin.reach_zono_approx(I, dis_opt)
            return R
