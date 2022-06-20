#!/usr/bin/python3
from random import seed
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/nn/funcs/satlin/")
# sys.path.insert(0, "engine/nn/fnn/FFNN/")
# from Operation import Operation
from poslin import PosLin
from satlin import SatLin


# @: matrix multi for np array (change later)
class Layer:
    """
        LAYERS is a class that contains only reachability analysis method using
        stars, this gonna replace Layer class in the future, i.e., delete
        polyhedron-base reachability analysis method

    Args:
        @W=np.array([]),  # weight_mat
        @b=np.array([]),  # bias vector
        @f='',  # activation function
        @N=0,  # number of neurons
        @gamma=0,  # used only for leakReLU layer
        @option='',  # parallel option, 'parallel' or '' '
        @dis_opt='',  # display option, 'display' or '' '
        @lp_solver='gurobi',  # lp solver option, 'gurobi'
        @relaxFactor=0  # use only for approx-star method

    Returns:
        Layer Object
    """

    # ------------- constructor, evaluation, sampling ------------
    def __init__(self, *args):
        """
        Constructor

        Raises:
            'error: Invalid number of input arguments, should be 3 or 4'
        """

        # ------------- Initialize Layer properties with empty numpy sets and zeros -------------
        [self.W, self.b] = [np.array([]), np.array([])]
        self.f = ''
        self.N = 0
        self.gamma = 0
        self.option = ''
        self.dis_opt = ''
        self.lp_solver = 'gurobi'
        self.relaxFactor = 0

        len_args = len(args)
        if len_args == 3:
            [self.W, self.b, self.f] = args
            self.check_essential_properties(self.W, self.b, self.f)
        elif len_args == 4:
            [self.W, self.b, self.f, self.gamma] = args
            self.check_essential_properties(self.W, self.b, self.f)
            self.check_gamma_for_LeakyReLU(self.gamma)
        elif len_args == 0:
            pass
        else:
            raise Exception(
                'error: Invalid number of input arguments, should be 3 or 4')

    def check_essential_properties(self, W, b, N):
        """
        Checks essential properties of Layer (W, b, N)

        return:
            @W=np.array([]),  # weight_mat
            @b=np.array([]),  # bias vector
            @N=0,  # number of neurons
        """

        # assert isinstance(W, np.array), 'error: weight_mat is not an np array'
        # assert isinstance(b, np.array), 'error: bias is not an np array'
        if W.shape[0] == b.shape[0]:
            # ------------- TODO: double precision here -------------
            # obj.W = double(W)
            # obj.b = double(b)
            self.N = W.shape[0]
        else:
            'error: Insonsistent dimensions between Weights matrix and bias vector'

    def check_gamma_for_LeakyReLU(self, gamma):
        """
        Checks essential propertie of Layer (gamma)
        
        Return:
            @gamma=0,  # used only for leakReLU layer

        """

        if gamma >= 1:
            'error: Invalid parameter for leakyReLu, gamma should be <= 1'

    def __repr__(obj):
        return "class: %s \nW: %s \nb: %s \nf: %s \nN: %s \ngamma: %s \noption: %s \ndis_opt: %s \nlp_solver: %s \nrelaxFactor: %s" \
               % (obj.__class__, obj.W, obj.b, obj.f, obj.N, obj.gamma, obj.option, obj.dis_opt, obj.lp_solver, obj.relaxFactor)\

    # ------------- Evaluation Method -------------
    def evaluate(self, x):
        """
        evaluation of this layer with a specific vector
        """

        #assert isinstance(x, np.array), 'error: x is not a np array'

        if x.shape[0] != self.W.shape[1] or x.shape[1] != 1:
            'error: invalid or inconsistent input vector'
        y1 = self.W * x + self.b
        y1_len = y1.shape[0]
        if self.f == 'poslin':
            for i in range(y1_len):
                y1[i] = max(0, y1[i])

        # ...... other functions
        else:
            'error: unknown or unsupported activation function'
        return y1

    def sample(self, V):
        """
        Evaluate the value of the layer output with a set of vertices
        """

        assert isinstance(V, np.ndarray), 'error: V is not a np array'
        n = V.shape[1]
        Y = []
        for j in range(n):
            y = self.evaluate(V[:, j])
            Y.append(y)
        return Y

    def reach(*args):
        """
        reachability analysis method

        Args:
            @I: an array of inputs (list)
            @method: 'exact-star' or 'approx-star' or 'approx-zono' or 'abs-dom', i.e., abstract domain (support
            later) or 'face-latice', i.e., face latice (support later)
            @option:  'parallel' use parallel computing
                      '[]' or not declared -> don't use parallel computing

        Returns:
            _type_: _description_
        """

        # ------------- parse inputs -------------
        if (len(args) == 7):  # 7 arguments
            [
                obj, I, method, obj.option, obj.relaxFactor, obj.dis_opt,
                obj.lp_solver
            ] = args
        elif (len(args) == 6):  # 6 arguments
            [obj, I, method, obj.option, obj.relaxFactor, obj.dis_opt] = args
        elif (len(args) == 5):  # 5 arguments
            [obj, I, method, obj.option, obj.relaxFactor] = args
            # Relax factor is only use for approx-star method
        elif (len(args) == 4):  # 4 arguments
            [obj, I, method, obj.option] = args
        elif (len(args) == 3):  # 3 arguments
            [obj, I, method] = args
        else:
            'error: Invalid number of input arguments (should be 2,3,4,5, or 6)'

        if method != 'exact-star' and method != 'approx-star' and method != 'approx-star-fast' and method != 'approx-zono' and method != 'abs-dom' and method != 'exact-polyhedron' and method != 'approx-star-split' and method != 'approx-star-no-split':
            'error: Unknown reachability analysis method'

        if method == 'exact-star' and obj.f != 'purelin' and obj.f != 'leakyrelu' and obj.f != 'poslin' and obj.f != 'satlin' and obj.f != 'satlins':
            method = 'approx-star'
            print(
                '\nThe current layer has {objf} activation function -> cannot compute exact reachable set for the current layer, we use approx-star method instead'
                .format(objf=obj.f))

        n = len(I)
        S = []
        W1 = obj.W
        b1 = obj.b
        f1 = obj.f
        gamma1 = obj.gamma

        # ------------- reachability analysis using star set -------------
        if obj.option == 'parallel':
            print('Will support parallel computing later')
        else:
            from star import Star
            from zono import Zono
            from box import Box
            for i in range(n):
                # do not support Polyhedron
                # affine mapping y = Wx + b
                if isinstance(I[i], Star) or isinstance(
                        I[i], Zono) or isinstance(I[i], Box):
                    I1 = I[i].affineMap(W1, b1)

                # ------------- add more supported function if needed -------------
                if f1 == 'poslin':
                    S.append(
                        PosLin.reach(I1, method, '', obj.relaxFactor,
                                     obj.dis_opt, obj.lp_solver))
                elif f1 == 'satlin':
                    S.append(
                        SatLin.reach(I1, method, '', obj.dis_opt,
                                     obj.lp_solver))
                else:
                    'error: Unsupported activation function, currently support purelin, poslin(ReLU), satlin, satlins, logsig, tansig, softmax'
                return S


# ------------- Unused Functions -------------
# def __init__(
#     obj,
#     W=np.array([]),  # weight_mat
#     b=np.array([]),  # bias vector
#     f='',  # activation function
#     N=0,  # number of neurons
#     gamma=0,  # used only for leakReLU layer
#     option='',  # parallel option, 'parallel' or '' '
#     dis_opt='',  # display option, 'display' or '' '
#     lp_solver='gurobi',  # lp solver option, 'gurobi'
#     relaxFactor=0  # use only for approx-star method
# ):

#     assert isinstance(W, np.ndarray), 'error: weight_mat is not an ndarray'
#     assert isinstance(b, np.ndarray), 'error: bias is not an ndarray'

#     if W.size and b.size and len(f):
#         obj.W = W
#         obj.b = b
#         obj.N = N
#         obj.option = option
#         obj.dis_opt = dis_opt
#         obj.lp_solver = lp_solver
#         obj.relaxFactor = relaxFactor
#         if gamma != 0:
#             obj.gamma = gamma
#     else:
#         'error: Invalid number of input arguments, should be 3 or 4'

#     if W.shape[0] == b.shape[0]:
#         # double precision here
#         # obj.W = double(W)
#         # obj.b = double(b)
#         obj.N = W.shape[0]
#     else:
#         'error: Insonsistent dimensions between Weights matrix and bias vector'

#     if gamma >= 1:
#         'error: Invalid parameter for leakyReLu, gamma should be <= 1'

#     obj.f = f
#     obj.gamma = gamma

#     return
#     raise Exception('error: failed to create Layers')

# # flattening a layer into a sequence of operation
# def flatten(obj, reachMethod):
#     # @reachMethod: reachability method
#     # @Ops: an array of operations for the reachability of
#     # the layer

#     Ops = []
#     O1 = Operation('AffineMap', obj.W, obj.b)

#     if obj.f == 'poslin':
#         if reachMethod == 'exact-star':
#             # O2[obj.N] = Operation
#             O2 = []
#             for i in range(obj.N):
#                 O2.append(Operation('PosLin_stepExactReach', i))
#         elif reachMethod == 'approx-star':
#             O2 = Operation('PosLin_approxReachStar')
#         elif reachMethod == 'approx-zono':
#             O2 = Operation('PosLin_approxReachZono')

#     Ops.append(O1)
#     Ops.append(O2)

#     return Ops
