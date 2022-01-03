import numpy as np

class FFNNOLD:
    # FeedForward Neural Network Class
    # author: Yuntao Li
    # date: 11/13/2021

    def __init__(obj,
                 Layers = np.matrix([]), # An array of Layers, eg, Layers = [L1 L2 ... Ln]
                 nL = 0, # number of Layers
                 nN = 0, # number of Neurons
                 nI = 0, # number of Inputs
                 nO = 0, # number of Outputs

                 # properties for each set computation

                 reachScheme = '', # reachable set computation scheme
                 numOfCores = 0, # number of cores (workers) using in computation
                 inputSet = np.matrix([]), # input set
                 reachSet = np.matrix([]), # reachable set of each layers
                 outputSet = np.matrix([]), # output reach set
                 reachTime = np.matrix([]), # computation time for each layers
                 totalReachTime = 0, # total computation time
                 n_ReLU_reduced = np.matrix([]), # number of ReLU stepReach operation is reduced
                 total_n_ReLU_reduced = 0 # total number of ReLU stepReach operation is reduced
                 ):

        assert isinstance(Layers, np.ndarray), 'error: Layers matrix is not an ndarray'
        assert isinstance(inputSet, np.ndarray), 'error: inputSet matrix is not an ndarray'
        assert isinstance(reachSet, np.ndarray), 'error: reachSet vector is not an ndarray'
        assert isinstance(outputSet, np.ndarray), ' error: outputSet is not an ndarray'
        assert isinstance(reachTime, np.ndarray), 'error: reachTime is not an ndarray'
        assert isinstance(n_ReLU_reduced, np.ndarray), 'error: n_ReLU_reduced is not an ndarray'

        from engine.nn.layers import Layers
        if Layers.size:
            nL = np.size(Layers, 1) # number of Layer
            for i in range(nL):
                L = Layers[i]
                if not isinstance(L, Layers): 'error: Element of Layers array is not a Layer object'

            # check consistency between layers
            for i in range(nL-1):
                if not np.size(Layers[i].W, 1) == np.size(Layers[i+1].W, 2):
                    'error: Inconsistent dimensions between Layer and Layer'

            obj.Layers = Layers
            obj.nL = nL # number of layers
            obj.nI = np.size(Layers[i].W, 2) # number of inputs
            obj.nO = np.size(Layers[nL].W, 1) # number of outputs

            nN = 0
            for i in range(nL):
                nN = nN + Layers[i].N
            obj.nN = nN # number of neurons

