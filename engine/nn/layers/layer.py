#!/usr/bin/python3
import numpy as np
from engine.nn.funcs.poslin import PosLin

# @: matrix multi for np array (change later)
class Layer:
    # LAYERS is a class that contains only reachability analysis method using
    # stars, this gonna replace Layer class in the future, i.e., delete
    # polyhedron-base reachability analysis method

    # author: Yuntao Li
    # date: 11/15/2021

    # ------------- constructor, evaluation, sampling ------------
    def __init__(obj,
                 W = np.matrix([]), # weight_mat
                 b = np.matrix([]), # bias vector
                 f = '', # activation function
                 N = 0, # number of neurons
                 gamma = 0, # used only for leakReLU layer

                 option = '', # parallel option, 'parallel' or '' '
                 dis_opt = '', # display option, 'display' or '' '
                 lp_solver = 'gurobi', # lp solver option, 'gurobi'
                 relaxFactor = 0 # use only for approx-star method
                 ):

        assert isinstance(W, np.ndarray), 'error: weight_mat is not an ndarray'
        assert isinstance(b, np.ndarray), 'error: bias is not an ndarray'

        if W.size and b.size and len(f):
            obj.W = W
            obj.b = b
            obj.N = N
            obj.option = option
            obj.dis_opt = dis_opt
            obj.lp_solver = lp_solver
            obj.relaxFactor = relaxFactor
            if gamma != 0:
                obj.gamma = gamma
        else: 'error: Invalid number of input arguments, should be 3 or 4'

        if W.shape[0] == b.shape[0]:
            # double precision here
            # obj.W = double(W)
            # obj.b = double(b)
            obj.N = W.shape[0]
        else: 'error: Insonsistent dimensions between Weights matrix and bias vector'

        if gamma >= 1: 'error: Invalid parameter for leakyReLu, gamma should be <= 1'

        obj.f = f
        obj.gamma = gamma

        return
        raise Exception('error: failed to create Layers')

    def __repr__(obj):
        return "class: %s \nW: %s \nb: %s \nf: %s \nN: %s \ngamma: %s \noption: %s \ndis_opt: %s \nlp_solver: %s \nrelaxFactor: %s" \
               % (obj.__class__, obj.W, obj.b, obj.f, obj.N, obj.gamma, obj.option, obj.dis_opt, obj.lp_solver, obj.relaxFactor)\

    # Evaluation Method
    def evaluate(obj, x): # evaluation of this layer with a specific vector
        assert isinstance(x, np.ndarray), 'error: x is not an ndarray'

        if x.shape[0] != obj.W.shape[1] or x.shape[1] != 1:
            'error: invalid or inconsistent input vector'
        y1 = obj.W * x + obj.b

        if obj.f == 'poslin':
            y = PosLin(y1)
        # ...... other functions
        else: 'error: unknown or unsupported activation function'
        return

    # Evaluate the value of the layer output with a set of vertices
    def sample(obj, V):
        assert isinstance(V, np.ndarray), 'error: V is not an ndarray'
        n = V.shape[1]
        Y = []
        for j in range(n):
            y = obj.evaluate(V[:, j])
            Y.append(y)
        return Y

    # ---------------- reachability analysis method ------------
    def reach(*args):
    # @I: an array of inputs (list)
    # @method: 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom', i.e., abstract domain (support
    # later) or 'face-latice', i.e., face latice (support later)
    # @option:  'parallel' use parallel computing
    #           '[]' or not declared -> don't use parallel computing
    # author: Yuntao Li

    # parse inputs
        if (len(args) == 7): # 7 arguments
            obj = args[0]
            I = args[1]
            method = args[2]
            obj.option = args[3]
            obj.relaxFactor = args[4] # only use for approx-star method
            obj.dis_opt = args[5]
            obj.lp_solver = args[6]
        elif (len(args) == 6): # 6 arguments
            obj = args[0]
            I = args[1]
            method = args[2]
            obj.option = args[3]
            obj.relaxFactor = args[4] # only use for approx-star method
            obj.dis_opt = args[5]
        elif (len(args) == 5): # 5 arguments
            obj = args[0]
            I = args[1]
            method = args[2]
            obj.option = args[3]
            obj.relaxFactor = args[4] # only use for approx-star method
        elif (len(args) == 4): # 4 arguments
            obj = args[0]
            I = args[1]
            method = args[2]
            obj.option = args[3]
        elif (len(args) == 3):  # 3 arguments
            obj = args[0]
            I = args[1]
            method = args[2]
        else: 'error: Invalid number of input arguments (should be 2,3,4,5, or 6)'

        if method != 'exact-star' and method != 'approx-star' and method != 'approx-star-fast' and method != 'approx-zono' and method != 'abs-dom' and method != 'exact-polyhedron'and method != 'approx-star-split' and method != 'approx-star-no-split':
            'error: Unknown reachability analysis method'

        if method == 'exact-star' and obj.f != 'purelin' and obj.f != 'leakyrelu' and obj.f != 'poslin' and obj.f != 'satlin' and obj.f != 'satlins':
            method = 'approx-star'
            print('\nThe current layer has {objf} activation function -> cannot compute exact reachable set for the current layer, we use approx-star method instead' .format(objf = obj.f))

        n = len(I)
        S = []
        W1 = obj.W
        b1 = obj.b
        f1 = obj.f
        gamma1 = obj.gamma

        if obj.option == 'parallel': # reachability analysis using star set
            print('Will support parallel computing later')
        else:
            from engine.set.star import Star
            from engine.set.zono import Zono
            from engine.set.box import Box
            for i in range(n):
                # do not support Polyhedron
                # affine mapping y = Wx + b
                if isinstance(I[i], Star):
                    I1 = I[i].affineMap(W1, b1);
                elif isinstance(I[i], Zono):
                    I1 = I[i].affineMap(W1, b1);
                elif isinstance(I[i], Box):
                    I1 = I[i].affineMap(W1, b1);

                # add more supported function if needed
                if f1 == 'poslin':
                    S.append(PosLin.reach(I1, method, [], obj.relaxFactor, obj.dis_opt, obj.lp_solver))
                else: 'error: Unsupported activation function, currently support purelin, poslin(ReLU), satlin, satlins, logsig, tansig, softmax'
                return S

    # flattening a layer into a sequence of operation...