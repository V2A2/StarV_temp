#!/usr/bin/python3
import numpy as np
import sys, copy

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/nn/funcs/satlin/")
sys.path.insert(0, "engine/set/star/")

from star import Star
from poslin import PosLin
from satlin import SatLin

class Operation:
    """
    Operation : class for specific operation in Feedforward neural network reachability

     An Operation can be:
       1) AffineMap
       2) PosLin_stepExactReach
       3) PosLin_approxReachZono
       4) PosLin_approxReachStar

       5) SatLin_approxReachStar
       6) SatLin_stepExactReach
       7) SatLin_approxReachZono

       8) SatLins_stepExactReach
       9) SatLins_approxReachStar
       10) SatLins_approxReachZono


    The main method for Operation class is:
      2) execute
    The Operation class is used for verification of FFNN using Deep First Search algorithm
    """

    def __init__(self, *args):
        """
        constructor

        Args:
            @name: name of the operation
            @W: affine mapping matrix
            @b: affine mapping vector
            @index: neuron index
        """

        self.Name = ''
        # used if the operation is an affine mapping operation
        self.map_mat = np.array([])
        # used if the operation is an affine mapping operation
        self.map_vec = np.array([])
        # used if the operation is PosLin or SatLin or SatLins stepReach operation
        # index is the index of the neuron the stepReach is performed
        self.index = np.array([])
        self.method = ''

        len_args = len(args)
        if len_args == 3:
            [name, mat, vec] = copy.deepcopy(args)
            if name != 'AffineMap':
                'error: The operation is not an affine mapping operation'

            if mat.shape[0] != vec.shape[0]:
                'error: Inconsistent dimension between that affine mapping matrix and vector'

            if vec.shape[0] != 1:
                'error: Affine mapping vector should have one row'

            self.Name = name
            self.map_mat = mat
            self.map_vec = vec

        elif len_args == 2:
            [name, index] = copy.deepcopy(args)
            if name != 'PosLin_stepExactReach' and name != 'SatLin_stepExactReach' and name != 'SatLins_stepExactReach':
                'error: Unknown operation name'
            if index < 1:
                'error: Invalid neuron index'

            self.Name = name
            self.index = index

        elif len_args == 1:
            [name] = copy.deepcopy(args)

            S1 = (name != 'PosLin_approxReachStar'
                  and name != 'PosLin_approxReachZono')
            S2 = (name != 'SatLin_approxReachStar'
                  and name != 'SatLin_approxReachZono')
            S3 = (name != 'SatLins_approxReachStar'
                  and name != 'SatLins_approxReachZono')

            if S1 and S2 and S3:
                'error: Unkown operation name'
            self.Name = name
        else:
            'error: Invalid number of arguments'

        return

    def __repr__(obj):
        return "class: %s \nName: %s \n" \
               % (obj.__class__, obj.Name)\

    def execute(obj, I):
        """
        execute the operation

        Args:
            @I: a star input set

        Returns:
            @S: a star output set or an array of star output sets
        """
        from star import Star
        # assert isinstance(I, Star), 'error: input set is not a star set'

        if obj.Name == 'AffineMap':
            S = I.affineMap(obj.map_mat, obj.map_vec)
            return S
        # PosLin
        elif obj.Name == 'PosLin_stepExactReach':
            # [xmin, xmax] = I.estimateRange(obj.index)
            # S = PosLin.stepReach(I, obj.index, xmin, xmax)
            S = PosLin.stepReach(I, obj.index)
            return S
        elif obj.Name == 'PosLin_approxReachStar':
            S = PosLin.reach_star_approx(I)
            return S
        elif obj.Name == 'PosLin_approxReachStar2':
            S = PosLin.reach_star_approx2(I)
            return S
        elif obj.Name == 'PosLin_approxReachZono':
            S = PosLin.reach_zono_approx(I)
            return S
        # SatLin
        elif obj.Name == 'SatLin_stepExactReach':
            S = SatLin.stepReach(I, obj.index)
            return S
        elif obj.Name == 'SatLin_approxReachStar':
            S = SatLin.reach_star_approx(I)
            return S
        elif obj.Name == 'SatLin_approxReachZono':
            S = SatLin.reach_zono_approx(I)
            return S
        else:
            'error: Unknown operation'
