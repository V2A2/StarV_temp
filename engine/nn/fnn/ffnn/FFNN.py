#!/usr/bin/python3
import sys
import copy
import numpy as np
import time

sys.path.insert(0, "engine/nn/layers/layer/")
sys.path.insert(0, "engine/set/imagestar/")
sys.path.insert(0, "engine/set/halfspace/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/set/zono/")
sys.path.insert(0, "engine/set/box/")

from layer import Layer
from star import Star
from box import Box
from zono import Zono
from imagestar import ImageStar
from halfspace import HalfSpace


# ------------- Constructor, evaluation, sampling, print method -------------
class FFNN:
    """
    FFNNS Class is a new feedforward network class used to replace the old FFNN in the future
    reachability analysis method: 'exact-star', 'approx-star'
    """

    def __init__(self, *args):
        """
        FFNN Constructor
        """

        # Initialize Star properties with empty numpy sets and zero values.
        self.Name = 'net'  # default is 'net'
        self.Layers = []  # An list of Layer, eg, Layer = [L1 L2 ... Ln]
        self.nL = 0  # number of Layer
        self.nN = 0  # number of Neurons
        self.nI = 0  # number of Input
        self.nO = 0  # number of Outputs

        # properties for each set computation
        self.reachMethod = 'exact-star'  # reachable set computation scheme, default - 'star'
        self.reachOption = ''  # parallel option, default - non-parallel computing
        self.relaxFactor = 0  # use only for approximate star method, 0 mean no relaxation
        self.numCores = 1  # number of cores (workers) using in computation
        self.inputSet = []  # input set
        self.reachSet = []  # reachable set of each Layer
        self.outputSet = []  # output reach set
        self.reachTime = []  # computation time for each Layer
        self.numReachSet = []  # number of reach sets over Layer
        self.totalReachTime = 0  # total computation time
        self.numSamples = 0  # default number of samples using to falsify a property
        self.unsafeRegion = []  # unsafe region of the network
        self.getCounterExs = 0  # default, not getting counterexamples

        self.Operations = []  # flatten a network into a sequence of operations

        self.dis_opt = ''  # display option
        self.lp_solver = 'gurobi'  # lp solver option, should be 'gurobi'

        len_args = len(args)
        if len_args == 1:
            [Layers] = copy.deepcopy(args)
            name = 'net'
        elif len_args == 2:
            [Layers, name] = copy.deepcopy(args)
        else:
            'error: Invalid number of inputs'

        # print("\n Layers ------------------------ \n", Layers)
        from layer import Layer
        nL = len(Layers)
        # print("\n nL ------------------------ \n", nL)
        for i in range(nL):
            L = Layers[i]
            assert isinstance(L, Layer), (
                'error: Element %d of Layers array is not a LayerS object' % i)

        # ------------- check consistency between layers -------------
        for i in range(nL - 1):
            assert Layers[i].W.shape[0] == Layers[i + 1].W.shape[1], (
                'error: Inconsistent dimensions between Layer %d and Layer %d'
                % (i, i + 1))

        self.Layers = Layers
        self.nL = nL  # number of Layer
        self.nI = Layers[0].W.shape[1]  # number of inputs
        self.nO = Layers[nL - 1].W.shape[0]  # number of outputs
        self.Name = name

        nN = 0
        for i in range(nL):
            nN = nN + Layers[i].N
        self.nN = nN  # number of neurons

    def __repr__(obj):
        return "class: %s \nName: %s \nLayers: %s \nnL: %s \nnN: %s \nnI: %s \nnO: %s \nreachMethod: %s \nreachOption: %s \nrelaxFactor: %s" \
               "\nnumCores: %s \ninputSet: %s \nreachSet: %s \noutputSet: %s \nreachTime: %s \nnumReachSet: %s \nnumSamples: %s " \
               "\nunsafeRegion: %s \ngetCounterExs: %s \nOperations: %s \ndis_opt: %s \nlp_solver: %s " \
               % (obj.__class__, obj.Name, obj.Layers, obj.nL, obj.nN, obj.nI, obj.nO, obj.reachMethod, obj.reachOption, obj.relaxFactor,
                  obj.numCores, obj.inputSet, obj.reachSet, obj.outputSet, obj.reachTime, obj.numReachSet, obj.numSamples, obj.unsafeRegion,
                  obj.getCounterExs, obj.Operations, obj.dis_opt, obj.lp_solver)

    def rand(neurons, funcs):
        """
        random generate a network for testing

        Args:
            neurons: an array of neurons of input layer - hidden layers - output layers
            funcs: an array of activation functions of hidden layers
        """

        n = len(neurons)

        if n < 2:
            'error: Network should have at least one input layer, zero hidden layer, and one output layer, i.e., length(neurons) >= 2'
        for i in range(n):
            if neurons[i] <= 0:
                'error: Invalid number of neurons at layer'
        
        m = len(funcs)
        for i in range(m):
            a = (funcs[i] != 'poslin' and funcs[i] != 'satlin')
            b = (funcs[i] != 'logsig' and funcs[i] != 'tansig')
            if a and b:
                'error: Unknown or unsupport activation function'

        if m != 1 and m != n-1:
            'error: Inconsistency between the number of layers and the number of activation functions'

        layers = []
        for i in range(1, n):
            W = np.random.rand(neurons[i], neurons[i-1]);
            b = np.random.rand(neurons[i]);

            if m != 1:
                L = Layer(W, b, funcs[i-1])
            else:
                L = Layer(W, b, funcs)

            layers.append(L)
        
        net = FFNN(layers)

        return net

    # Evaluation of a FFNN
    def evaluate(self, x):
        """
         Evaluation of a FFNN

        Args:
            @x: input vector x
        
        Returns:
            @y: output vector y
        """

        y = x
        for i in range(self.nL):
            y = self.Layers[i].evaluate(y)
        return y

    # Sample of FFNN
    def sample(obj, V):
        """
        sample the output of each layer in the FFNN based on the vertices of input set I, this is useful for testing

        Args:
            @V: array of vertices to evaluate

        Returns:
            @Y: output which is a cell array
        """

        In = V
        if V.size:
            # print("\n In ------------------------ \n", In)
            for i in range(obj.nL):
                In = obj.Layers[i].sample(In)
        return In

    def isPieceWiseNetwork(obj):
        """
        check if all activation functions are piece-wise linear
        """

        n = obj.nL
        b = 1
        for i in range(obj.nL):
            f = obj.Layer[i].f
            if f != 'poslin' and f != 'purelin' and f != 'satlin' and f != 'satlins' and f != 'leakyrelu':
                b = 0
                return

    # print information to a file
    def printtoFile(obj, file_name):
        """
        print information to a file

        Args:
            @file_name: name of file you want to store all data information
        """

        f = open(file_name, 'w')
        f.write('Feedforward Neural Network Information\n')
        f.write('\nNumber of Layer: %d' % obj.nL)
        f.write('\nNumber of neurons: %d' % obj.nN)
        f.write('\nNumber of inputs: %d' % obj.nI)
        f.write('\nNumber of outputs: %d' % obj.nO)

        if obj.reachSet.size:
            f.write('\n\nReach Set Information')
            f.write('\nReachability method: %s' % obj.reachMethod)
            f.write('\nNumber of cores used in computation: %d' % obj.numCores)

            for i in range(len(obj.reachSet) - 1):
                f.write(
                    '\nLayer %d reach set consists of %d sets that are computed in %.5f seconds'
                    % (i, obj.numReachSet[i], obj.reachTime[i]))

            f.write(
                '\nOutput Layer reach set consists of %d sets that are computed in %.5f seconds'
                % (obj.numReachSet(obj.nL), obj.reachTime(obj.nL)))
            f.write('\nTotal reachable set computation time: %.5f' %
                    obj.totalReachTime)

        f.close()

    def printtoConsole(obj):
        """
        print information to console
        """

        print('Feedforward Neural Network Information\n')
        print('\nNumber of Layer: %d' % obj.nL)
        print('\nNumber of neurons: %d' % obj.nN)
        print('\nNumber of inputs: %d' % obj.nI)
        print('\nNumber of outputs: %d' % obj.nO)

        if obj.reachSet.size:
            print('\n\nReach Set Information')
            print('\nReachability method: %s' % obj.reachMethod)
            print('\nNumber of cores used in computation: %d' % obj.numCores)

            for i in range(len(obj.reachSet) - 1):
                print(
                    '\nLayer %d reach set consists of %d sets that are computed in %.5f seconds'
                    % (i, obj.numReachSet[i], obj.reachTime[i]))

            print(
                '\nOutput Layer reach set consists of %d sets that are computed in %.5f seconds'
                % (obj.numReachSet(obj.nL), obj.reachTime(obj.nL)))
            print('\nTotal reachable set computation time: %.5f' %
                  obj.totalReachTime)

# ------------- Reachability analysis method and Verification method -------------

    def reach(*args):
        """
        Reachability analysis

        Args:
            @I: input set, a star set
            @method: = 'exact-star' or 'approx-star' -> compute reach set using stars
                       'abs-dom' -> compute reach set using abstract
                       'approx-zono' -> compute reach set using zonotope
            @numOfCores: number of cores you want to run the reachable set computation

        Returns:
            @R: output set
            @t : computation time
        """
        len_args = len(args)
        if len_args == 7:
            obj = args[0]
            obj.inputSet = args[1]  # input set
            obj.reachMethod = args[2]  # reachability analysis method
            obj.numCores = args[3]  # number of cores used in computation
            obj.relaxFactor = args[4]  # used only for approx-star method
            obj.dis_opt = args[5]
            obj.lp_solver = args[6]
        elif len_args == 6:
            obj = args[0]
            obj.inputSet = args[1]  # input set
            obj.reachMethod = args[2]  # reachability analysis method
            obj.numCores = args[3]  # number of cores used in computation
            obj.relaxFactor = args[4]
            obj.dis_opt = args[5]
        elif len_args == 5:
            obj = args[0]
            obj.inputSet = args[1]  # input set
            obj.reachMethod = args[2]  # reachability analysis method
            obj.numCores = args[3]  # number of cores used in computation
            obj.relaxFactor = args[4]
        elif len_args == 4:
            obj = args[0]
            obj.inputSet = args[1]  # input set
            obj.reachMethod = args[2]  # reachability analysis method
            obj.numCores = args[3]  # number of cores used in computation
        elif len_args == 3:
            obj = args[0]
            obj.inputSet = args[1]  # input set
            obj.reachMethod = args[2]  # reachability analysis method
            obj.numCores = 1  # number of cores used in computation
        elif len_args == 2:
            obj = args[0]
            arg1 = copy.deepcopy(args[1])
            if not isinstance(arg1, np.ndarray):
                obj.inputSet = arg1  # input set
                obj.reachMethod = 'exact-star'  # reachability analysis method
                obj.numCores = 1  # number of cores used in computation
            else:
                if 'inputSet' in arg1.dtype.names:
                    obj.inputSet = arg1[0]['InputSet']
                if 'numCores' in arg1.dtype.names:
                    obj.numCores = arg1[0]['numCores']
                if 'reachMethod' in arg1.dtype.names:
                    obj.reachMethod = arg1[0]['reachMethod']
                if 'dis_opt' in arg1.dtype.names:
                    obj.dis_opt = arg1[0]['dis_opt']
                if 'relaxFactor' in arg1.dtype.names:
                    obj.relaxFactor = arg1[0]['relaxFactor']
                if 'lp_solver' in arg1.dtype.names:
                    obj.lp_solver = arg1[0]['lp_solver']
        else:
            'error: Invalid number of input arguments (should be 1, 2, 3, 4, 5, or 6)'

        # ------------- if reachability analysis method is an over-approximate -------------
        # ------------- method, we use 1 core for computation -------------
        if obj.reachMethod != 'exact-star':
            obj.numCores = 1

        # ------------- Zonotope method accepts both star and zonotope input set -------------
        if obj.reachMethod == 'approx-zono' and isinstance(args[1], Star):
            obj.inputSet = args[1].getZono

        if obj.numCores == 1:
            obj.reachOption = ''  # don't use parallel computing
        else:
            print('working on this......')

        obj.reachSet = []
        obj.numReachSet = []
        obj.reachTime = []

        # ------------- compute reachable set -------------
        In = [obj.inputSet]

        for i in range(obj.nL):
            if obj.dis_opt == 'display':
                print('\n Computing reach set for Layer %d ... \n' % i)

            st = tic()
            #In = obj.Layer[i].reach(In, obj.reachMethod, obj.reachOption, obj.relaxFactor, obj.dis_opt, obj.lp_solver)
            In = obj.Layers[i].reach(In, obj.reachMethod, obj.reachOption,
                                     obj.relaxFactor, obj.dis_opt,
                                     obj.lp_solver)
            t1 = toc()

            obj.numReachSet = len(In)
            obj.reachTime.append(t1)

            if obj.dis_opt == 'display':
                print('\n Exact computation time: %.5f seconds \n' % t1)
                print(
                    '\n Number of reachable set at the output of layer %d: %d \n'
                    % (i, len(In)))

        obj.outputSet = In
        obj.totalReachTime = np.sum(obj.reachTime)
        S = obj.outputSet
        t = obj.totalReachTime

        if obj.dis_opt == 'display':
            print('\n Total reach set computation time: %.5f seconds \n' %
                  obj.totalReachTime)
            print('\n Total number of output reach sets: %d \n' %
                  len(obj.outputSet))

        return [S, t]

    def verify(*args):
        """
        Verification method

        Args:
            @I: input set, need to be a star set
            @U: unsafe region, a HalfSpace
            @method: = 'star' -> compute reach set using stars
            @numOfCores: number of cores you want to run the reachable
                set computation, @numOfCores >= 1, maximum is the number of
                cores in your computer.
            @n_samples : number of simulations used for falsification if
                using over-approximate reachability analysis, i.e.,
                'approx-zono'
            note: n_samples = 0 -> do not do falsification

        Returns:
            @safe: = 1-> safe, 0-> unsafe, 2 -> unknown
            @vt: verification time
            @counterExamples: counterexamples
        """
        len_args = len(args)
        if len_args == 2:
            obj = args[0]
            arg1 = copy.deepcopy(args[1])
            if not isinstance(arg1, np.ndarray):
                obj.inputSet = arg1  # input set
                obj.reachMethod = 'exact-star'  # reachability analysis method
                obj.numCores = 1  # number of cores used in computation
            else:
                if 'inputSet' in arg1.dtype.names:
                    obj.inputSet = arg1[0]['InputSet']
                if 'unsafeRegion' in arg1.dtype.names:
                    obj.unsafeRegion = arg1[0]['unsafeRegion']
                if 'numCores' in arg1.dtype.names:
                    obj.numCores = arg1[0]['numCores']
                if 'reachMethod' in arg1.dtype.names:
                    obj.reachMethod = arg1[0]['reachMethod']
                if 'dis_opt' in arg1.dtype.names:
                    obj.dis_opt = arg1[0]['dis_opt']
                if 'relaxFactor' in arg1.dtype.names:
                    obj.relaxFactor = arg1[0]['relaxFactor']
                if 'lp_solver' in arg1.dtype.names:
                    obj.lp_solver = arg1[0]['lp_solver']
        elif (len_args == 3):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
        elif (len_args == 4):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
        elif (len_args == 5):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
        elif (len_args == 6):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.numSamples = args[5]
        elif (len_args == 7):
            obj = args[0]
            obj.inputSet = args[1]
            obj.unsafeRegion = args[2]
            obj.reachMethod = args[3]
            obj.numCores = args[4]
            obj.numSamples = args[5]
            obj.getCounterExs = args[6]
        else: 'error: Invalid number of inputs, should be 2 3 4 5 6 7'

        if obj.numSamples > 0:
            tic_falsify = tic()
            print('\nPerform fasification with %d random simulations' % obj.numSamples)
            counterExamples = obj.falsify(obj.inputSet, obj.unsafeRegion, obj.numSamples)
            toc_falsify = toc()
        else:
            counterExamples = []

        if len(counterExamples) != 0:
            safe = 0
            obj.outputSet = []
            vt = toc_falsify
        else:
            if obj.numCores > 1:
                print('\n working on this..... \n')
            t = tic()
            # perform reachability analysis
            [R, time] = obj.reach(obj.inputSet, obj.reachMethod, obj.numCores)

            if obj.reachMethod == 'exact-star':
                n = len(R)
                counterExamples = []
                safes = []
                G = obj.unsafeRegion[0].G
                g = obj.unsafeRegion[0].g
                getCEs = obj.getCounterExs
                V = obj.inputSet.V

                if obj.numCores > 1:
                    print('working on this....')
                else:
                    safe = 1
                    for i in range(n):
                        R1 = copy.deepcopy(R[i])
                        [H1, empty] = R1.intersectHalfSpace(obj.unsafeRegion[0].G, obj.unsafeRegion[0].g)
                        if empty == 0:
                            if obj.getCounterExs and obj.reachMethod == 'exact-star':
                                counterExamples.append(Star(obj.inputSet.V, R[i].C, R[i].d, R[i].predicate_lb, R[i].predicate_ub))
                            else:
                                safe = 0
                                break
            else:
                safe = 1
                if obj.reachMethod != 'exact-polyhedron':
                    if obj.reachMethod == 'approx-zono':
                        R = R.toStar
                    if R.intersectHalfSpace(obj.unsafeRegion[0].G, obj.unsafeRegion[0].g).size:
                        safe = 1
                    else:
                        safe = 2
                else:
                    n = len(R)
                    for i in range (n):
                        R1 = copy.deepcopy(R[i])
                        [H1, empty] = R1.intersectHalfSpace(obj.unsafeRegion[0].G, obj.unsafeRegion[0].g)
                        if empty == 0:
                            safe = 0
                            break
            vt = toc()
        return [safe, vt, counterExamples]


# ------------- checking safety method or falsify safety property -------------

    def falsify(obj, I, U, n_samples):
        """
        falsify safety property using random simulation

        Args:
            @input: star set input
            @U: unsafe region(s), a set of HalfSpaces
            @n_samples: number of samples used in falsification
            @counter_inputs: counter inputs that falsify the property

        Returns:
            @counter_inputs
        """

        if isinstance(I, Zono) or isinstance(I, Box):
            I1 = I.toStar
        elif isinstance(I, Star):
            I1 = I
        else:
            'error: Unknown set representation'

        m = len(U)
        for i in range(m):
            if not isinstance(U[i], HalfSpace):
                ('error: %d^th unsafe region is not a HalfSpace' % i)

        if n_samples < 1:
            'error: Invalid number of samples'

        V = I1.sample(n_samples)
        # print("\n V ------------------------ \n", V)
        # print("\n V ------------------------ \n", V[:, 0])
        n = V.shape[1]  # number of samples
        # print("\n n ------------------------ \n", n)
        counter_inputs = np.array([])

        for i in range(n):
            V_vector = V[:, i].reshape(-1, 1)
            y = obj.evaluate(V_vector)
            # print("\n y ------------------------ \n", y)
            # print("\n V_vector ------------------------ \n", V_vector)
            for j in range(m):
                assert isinstance(
                    U[j], HalfSpace), 'error: U is not a HalfSpace object'
                if y.size and U[j].contains(y):
                    if counter_inputs.size:
                        counter_inputs = np.column_stack(
                            [counter_inputs, V_vector])
                        # print("\n counter_inputs ------------------------ \n", counter_inputs)
                    else:
                        counter_inputs = V_vector
                        # print("\n counter_inputs ------------------------ \n", counter_inputs)
                    # counter_inputs.extend(V_vector)

        return counter_inputs

    def isSafe(*args):
        """
        checking safety method

        Args:
            @I: input set, need to be a star set
            @U: unsafe region, a set of HalfSpaces
            @method: = 'star' -> compute reach set using stars
            @numOfCores: number of cores you want to run the reachable set computation
            @n_samples: number of simulations used for falsification if
                        using over-approximate reachability analysis, i.e.,
                        'approx-zono'
                        note: n_samples = 0 -> do not do falsification

        Returns:
            @safe: = 1 -> safe, = 0 -> unsafe, = 2 -> uncertain
            @t : verification time
            @counter_inputs
        """

        # parse inputs
        len_args = len(args)
        if len_args == 6:
            obj = args[0]  # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = args[4]
            numOfCores = args[5]
        elif len_args == 5:
            obj = args[0]  # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = args[4]
            numOfCores = 1
        elif len_args == 4:
            obj = args[0]  # FFNNS objects
            I = args[1]
            U = args[2]
            method = args[3]
            n_samples = 1000
            numOfCores = 1
        elif len_args == 3:
            obj = args[0]  # FFNNS objects
            I = args[1]
            U = args[2]
            method = 'exact-star'
            n_samples = 0
            numOfCores = 1
        else:
            'error: Invalid number of input arguments, should be 3, 4, 5, 6'

        t_start = tic()

        if len(U):
            'error: Please specify unsafe region using Half-space class'

        # performing reachability analysis
        [R, time] = obj.reach(I, method, numOfCores)
        # print("\n R ------------------------ \n", R.__repr__())

        # check safety
        n = len(R)
        # print("\n n ------------------------ \n", n)
        m = len(U)
        # print("\n m ------------------------ \n", m)
        R1 = []
        for i in range(n):
            if isinstance(R[i], Zono):
                B = R[i].getBox
                R1.append(B.toStar)  # transform to star sets
            else:
                R1.append(R[i])
        # print("\n R1 ------------------------ \n", R1.__repr__())

        violate_inputs = []
        if numOfCores == 1:
            for i in range(n):
                for j in range(m):
                    [S, empty] = R1[j].intersectHalfSpace(U[j].G, U[j].g)
                    # print("\n S ------------------------ \n", S.__repr__())
                    if empty == 0 and method == 'exact-star':
                        I1 = Star(I.V, S.C, S.d)  # violate input set
                        violate_inputs.append(I1)
                    else:
                        violate_inputs.append(S)
        elif numOfCores > 1:
            print('\n Working on this...... \n')

        if len(violate_inputs) == 0:
            safe = 1
            counter_inputs = []
            print('\n The Network is Safe \n')
        else:
            if method == 'exact-star':
                safe = 0
                counter_inputs = violate_inputs  # exact-method return complete counter input set
                print(
                    '\n The Network is unsafe, counter inputs contains %d stars \n'
                    % len(counter_inputs))
            else:
                if n_samples == 0:
                    print(
                        '\n Do not do falsification since n_samples = 0, you can choose to do falsification by set n_samples value > 0 \n'
                    )
                    safe = 2
                    counter_inputs = []
                else:
                    counter_inputs = obj.falsify(I, U, n_samples)
                    if len(counter_inputs) == 0:
                        safe = 2
                        print(
                            '\n Safety is uncertain under using %d samples to falsify the network \n'
                            % n_samples)
                        print(
                            '\n You can try to increase the samples for finding counter inputs \n'
                        )
                    else:
                        safe = 0
                        print(
                            '\n The network is unsafe, %d counter inputs are found using %d simulations \n'
                            % (len(counter_inputs), n_samples))

        t_end = toc()
        return [safe, t_end, counter_inputs]


# ------------- Verify FFNN with BFS, DFS, F-DFS -------------

    def start_pool(obj, numCores=1):
        """
        start parallel pool for computing

        Args:
            @numCores: Defaults to 1.
        """

        # if (len(args) == 1):
        #     obj = args[0]
        #     nCores = obj.numCores
        # elif (len(args) == 2):
        #     obj = args[0]
        #     nCores = args[1]
        if numCores:
            nCores = numCores
        else:
            'error: Invalid number of input arguments'

        if nCores > 1:
            print('Working on this........')
        return

    def flatten(obj, reachMethod):
        """
        flatten a FFNN into a sequence of operations for reachability

        Args:
            @reachMethod: reachability method
        """

        assert isinstance(obj, FFNN), 'error: obj is not an FFNN object'
        Ops = []
        for i in range(obj.nL):
            assert isinstance(obj.Layers[i],
                              Layer), 'error: obj is not a Layer object'
            Op = obj.Layers[i].flatten(reachMethod)
            # print("\n Op ------------------------ \n", Op)
            Ops += Op
            # print("\n Ops ------------------------ \n", Ops)
        obj.Operations = Ops

    def verify_F_DFS_SingleCore(*args):
        """
        F-DFS method to verify FFNN

        Args:
            @inputSets: a star set
            @unsafeRegion: a HalfSpace object
            @numCores: number of cores used for verification
            
        Returns:    
            @safe:  = 'safe' or 'unsafe' or 'unknown'
            @CEx: counter examples
        """

        len_args = len(args)
        assert np.mod(len_args, 2) != 0, 'error: Invalid number of arguments'

        obj = args[0]
        assert isinstance(obj, FFNN), 'error: obj is not an FFNN object'

        for i in range(1, len_args - 1):
            if np.mod(i, 2) != 0:
                if args[i] == 'InputSet':
                    obj.inputSet = args[i + 1]
                elif args[i] == 'UnsafeRegion':
                    obj.unsafeRegion = args[i + 1]
                elif args[i] == 'ReachMethod':
                    obj.reachMethod = args[i + 1]
                elif args[i] == 'NumCores':
                    obj.numCores = args[i + 1]
        # print("\n obj.inputSet ------------------------ \n", obj.inputSet.__repr__())
        obj.flatten(obj.reachMethod)
        # print("\n obj ------------------------ \n", obj.__repr__())
        N = len(obj.Operations)
        U = obj.unsafeRegion

        if obj.numCores > 1:
            print('Only support one core right now \n')
        else:
            S = np.array([(obj.inputSet, 0)],
                         dtype=[('data', (Star)), ('opsIndex', np.int32)])
            safe = 'safe'
            CEx = []
            while safe == 'safe' and S.size:
                # print("\n S ------------------------ \n", S)
                S1 = S[0]['data']
                id = S[0]['opsIndex']
                # print("\n Operations ------------------------ \n", obj.Operations[id])
                # print("\n S1 ------------------------ \n", S1)
                if id < N - 1:
                    if isinstance(S1, Star):
                        S2 = obj.Operations[id].execute(S1)
                    else:
                        S2 = obj.Operations[id].execute(S1[0])
                    # print("\n S2 ------------------------ \n", S2)
                    if isinstance(S2, Star):
                        S4 = np.array([(S2, id + 1)],
                                      dtype=[('data', (Star)),
                                             ('opsIndex', np.int32)])
                        # print("\n S4 ------------------------ \n", S4)
                        S = S[1:]
                        S = np.hstack([S4, S])
                        # print("\n S new ------------------------ \n", S)
                    else:
                        if len(S2) == 3:
                            # print("\n 3 ------------------------ \n")
                            S3_0 = np.array([(S2[0], id + 1)],
                                            dtype=[('data', (Star)),
                                                   ('opsIndex', np.int32)])
                            S3_1 = np.array([(S2[1], id + 1)],
                                            dtype=[('data', (Star)),
                                                   ('opsIndex', np.int32)])
                            S3_2 = np.array([(S2[2], id + 1)],
                                            dtype=[('data', (Star)),
                                                   ('opsIndex', np.int32)])
                            S3 = np.hstack([S3_0, S3_1, S3_2])
                            S = S[1:]
                            S = np.hstack([S3, S])
                            # print("\n S ------------------------ \n", S)
                        elif len(S2) == 2:
                            S3_0 = np.array([(S2[0], id + 1)],
                                            dtype=[('data', (Star)),
                                                   ('opsIndex', np.int32)])
                            S3_1 = np.array([(S2[1], id + 1)],
                                            dtype=[('data', (Star)),
                                                   ('opsIndex', np.int32)])
                            S3 = np.hstack([S3_0, S3_1])
                            S = S[1:]
                            S = np.hstack([S3, S])
                            # print("\n S ------------------------ \n", S)
                        elif len(S2) == 1:
                            S4 = np.array([(S2, id + 1)],
                                          dtype=[('data', (Star)),
                                                 ('opsIndex', np.int32)])
                            # print("\n S4 ------------------------ \n", S4)
                            S = S[1:]
                            S = np.hstack([S4, S])
                            # print("\n S new ------------------------ \n", S)

                # ------------------------ checking safety of the leaf sets ------------------------
                else:
                    # print("\n S1 ------------------------ \n", S1)
                    if isinstance(S1, Star):
                        S2 = obj.Operations[id].execute(S1)
                    else:
                        S2 = obj.Operations[id].execute(S1[0])
                    # print("\n S2 ------------------------ \n", S2)
                    if isinstance(S2, Star):
                        [H, empty] = S2.intersectHalfSpace(U.G, U.g)
                        if empty == 0:
                            if H.V.size:
                                if obj.reachMethod == 'exact-star':
                                    safe = 'unsafe'
                                    CEx = Star(obj.inputSet.V, H.C, H.d,
                                               H.predicate_lb, H.predicate_ub)
                                else:
                                    safe = 'unknown'
                                    CEx = []
                    else:
                        n = len(S2)
                        for i in range(n):
                            [H, empty] = S2[i].intersectHalfSpace(U.G, U.g)
                            if empty == 0:
                                if H.V.size:
                                    if obj.reachMethod == 'exact-star':
                                        safe = 'unsafe'
                                        CEx.append(
                                            Star(obj.inputSet.V, H.C, H.d,
                                                 H.predicate_lb,
                                                 H.predicate_ub))
                                    else:
                                        safe = 'unknown'
                                        CEx = []
                        # simplified
                        # if len(S2) == 3:
                        #     [H0, empty0] = S2[0].intersectHalfSpace(U.G, U.g)
                        #     [H1, empty1] = S2[1].intersectHalfSpace(U.G, U.g)
                        #     [H2, empty2] = S2[2].intersectHalfSpace(U.G, U.g)
                        #     if empty0 == 0:
                        #         if H0.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H0.C, H0.d,
                        #                            H0.predicate_lb,
                        #                            H0.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []
                        #     elif empty1 == 0:
                        #         if H1.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H1.C, H1.d,
                        #                            H1.predicate_lb,
                        #                            H1.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []
                        #     elif empty2 == 0:
                        #         if H2.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H2.C, H2.d,
                        #                            H2.predicate_lb,
                        #                            H2.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []
                        # elif len(S2) == 2:
                        #     [H0, empty0] = S2[0].intersectHalfSpace(U.G, U.g)
                        #     [H1, empty1] = S2[1].intersectHalfSpace(U.G, U.g)
                        #     if empty0 == 0:
                        #         if H0.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H0.C, H0.d,
                        #                            H0.predicate_lb,
                        #                            H0.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []
                        #     elif empty1 == 0:
                        #         if H1.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H1.C, H1.d,
                        #                            H1.predicate_lb,
                        #                            H1.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []
                        # elif len(S2) == 1:
                        #     [H, empty] = S2[0].intersectHalfSpace(U.G, U.g)
                        #     if empty == 0:
                        #         if H.V.size:
                        #             if obj.reachMethod == 'exact-star':
                        #                 safe = 'unsafe'
                        #                 CEx = Star(obj.inputSet.V, H.C, H.d,
                        #                            H.predicate_lb,
                        #                            H.predicate_ub)
                        #             else:
                        #                 safe = 'unknown'
                        #                 CEx = []

                    S = S[1:]
        return [safe, CEx]

    def verify_BFS_SingleCore(*args):
        """
        BFS method to verify FFNN

        Args:
            @inputSets: a star set
            @unsafeRegion: a HalfSpace object
            @numCores: number of cores used for verification
            
        Returns:    
            @safe:  = 'safe' or 'unsafe' or 'unknown'
            @CEx: counter examples
        """

        len_args = len(args)
        assert np.mod(len_args, 2) != 0, 'error: Invalid number of arguments'

        obj = args[0]
        assert isinstance(obj, FFNN), 'error: obj is not an FFNN object'

        for i in range(1, len_args - 1):
            if np.mod(i, 2) != 0:
                if args[i] == 'InputSet':
                    obj.inputSet = args[i + 1]
                elif args[i] == 'UnsafeRegion':
                    obj.unsafeRegion = args[i + 1]
                elif args[i] == 'ReachMethod':
                    obj.reachMethod = args[i + 1]
                elif args[i] == 'NumCores':
                    obj.numCores = args[i + 1]

        Rs = []
        N = obj.nL
        In = [obj.inputSet]

        for i in range(N):
            In = obj.Layers[i].reach(In, obj.reachMethod, obj.reachOption)
            Rs += In

        obj.outputSet = In
        R = obj.outputSet
        n = len(R)

        safe = 'safe'
        CEx = []
        G = obj.unsafeRegion.G
        g = obj.unsafeRegion.g

        for i in range(n):
            R1 = R[i]
            [H, empty] = R1.intersectHalfSpace(G, g)
            if empty == 0:
                if H.V.size:
                    if obj.reachMethod == 'exact-star':
                        safe = 'unsafe'
                        CEx += Star(obj.inputSet.V, H.C, H.d, H.predicate_lb,
                                    H.predicate_ub)
                    else:
                        safe = 'unknown'
                        CEx = []

        return [safe, CEx]






# ------------- Timing Helper function -------------
def TicTocGenerator():
    """
    Generator that returns time differences

    Returns: 
        the time difference
    """

    ti = 0  # initial time
    tf = time.time()  # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf - ti  # returns the time difference

TicToc = TicTocGenerator()  # create an instance of the TicTocGen generator

def toc(tempBool=True):
    """
    This will be the main function through which we define both tic() and toc()

    Args:
        tempBool (bool, optional): Defaults to True.

    Returns:
        the time difference yielded by generator instance TicToc
    """

    tempTimeInterval = next(TicToc)
    if tempBool:
        # print("Elapsed time: %f seconds.\n" % tempTimeInterval)
        return tempTimeInterval

def tic():
    """
    Records a time in TicToc, marks the beginning of a time interval
    """

    # ------------- Records a time in TicToc, marks the beginning of a time interval -------------
    toc(False)
