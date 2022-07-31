import numpy as np
import torch
import torch.nn as nn 
from eagerpy.framework import sign
from tensorflow.python.distribute.device_util import current
import sys

SIGN_POLAR_ZERO_POS_ONE = 'polar_zero_to_pos_one'
SIGN_NONNEGATIVE_ZERO_POS_ONE = 'nonnegative_zero_to_pos_one'

SIGN_REACH_ARGS_INPUT_IMAGES_ID = 0
SIGN_REACH_ARGS_METHOD_ID = 1
SIGN_REACH_ARGS_MODE_ID = 2

SIGN_RMI_EXACT_ARGS_INPUT_ID = 0
SIGN_RMI_EXACT_ARGS_MODE_ID = 1

SIGN_REACH_STAR_EXACT_INPUT_ID = 0
SIGN_REACH_STAR_EXACT_MODE_ID = 1

SIGN_REACH_STAR_APPROX_INPUT_ID = 0
SIGN_REACH_STAR_APPROX_MODE_ID = 1

SIGN_STEPREACH_ARGS_INPUT_ID = 0
SIGN_STEPREACH_ARGS_INDEX_ID = 1
SIGN_STEPREACH_ARGS_MODE_ID = 2

SIGN_DEFAULT_BOUNDS_INIT = {
        SIGN_POLAR_ZERO_POS_ONE : np.array([-1, 1]),
        SIGN_NONNEGATIVE_ZERO_POS_ONE : np.array([0, 1]),
    }

sys.path.insert(0, "engine/set/star")
from star import *

class Sign:
    """
        Sign class contains method for reachability analysis for Layer with
        Sign activation function
    """
    
    sign_bounds = SIGN_DEFAULT_BOUNDS_INIT
    
    @staticmethod
    def evaluate(input, mode):
        """
            Evaluates the layer using the given input
            
            input : np.array([*]) -> multi-dimensional array
            mode : string -> sign function type
            
            returns the results of applying Sign to the given input
        """
        if not isinstance(input, torch.FloatTensor):
            input = torch.FloatTensor(input)
        
        y = torch.sign(input).cpu().detach().numpy()
        
        if mode == SIGN_POLAR_ZERO_POS_ONE:
            y = y + (y == 0)
        elif mode == SIGN_NONNEGATIVE_ZERO_POS_ONE:
            y = y + (y == 0)
            y = y + (y == -1)
        else:
            raise Exception(SIGN_ERRMSG_MODE_NOT_SUPPORTED)
            
        return y
    
    @staticmethod
    def stepReach(*args):
        """
            Performs a stepReach operation on the given input
            
            input : Star -> input Star set
            index : index of the neuron performing stepSign
            mode : string -> sign function type
            
            returns a reachable set for the given inputg
        """
        result = []
        
        input = args[SIGN_STEPREACH_ARGS_INPUT_ID]
        index = args[SIGN_STEPREACH_ARGS_INDEX_ID]
        mode = args[SIGN_STEPREACH_ARGS_MODE_ID]
        
        assert isinstance(input, Star), 'error: %s' % SIGN_ERRORMSG_INVALID_INPUT
        
        xmin = input.getMin(index)
        xmax = input.getMax(index)
        
        sign_lb, sign_ub = Sign.get_sign_bounds(mode)
        sign = sign_lb
        
        if xmin == xmax and ((amin == sign_lb) or (xmin == sign_ub)):
            return input
        elif xmin >= sign_lb:
            sign = sign + sign_ub - sign_lb
            
        #new_V = input.get_V()
        new_V = input.V
        new_V[index, 0] = sign
        
        #new_predicate_lb = input.get_pred_lb()
        #new_predicate_ub = input.get_pred_ub()
        new_predicate_lb = input.predicate_lb
        new_predicate_ub = input.predicate_ub
        
        new_predicate_lb[index] = sign
        new_predicate_ub[index] = sign
        new_V[index, 1 : new_V.shape[1]] = np.zeros((1, new_V.shape[1] - 1))
        
        #new_C = input.get_C()
        #new_d = input.get_d()
        new_C = input.C
        new_d = input.d
        
        #input.get_Z()
        if not input.Z.isempty():
            #current_c = input.get_Z().get_c()
            current_c = input.Z.c
            current_c[index] = sign
            #V1 = input.get_Z().get_V()
            V1 = input.Z.V
            
            V1[index, 1 : V1.shape[1]] = np.zeros((1, V1.shape[1] - 1))
            
            new_Z = Zono(current_c, V1)
        else:
            new_Z = []
            
        S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z)
        
        result.append(S1)
        
        if sign != sign_ub and xmax > 0:
            sign = 1
            new_V[index, 0] = sign
            
            new_predicate_lb[index] = sign
            new_predicate_ub[index] = sign
            
            # input.get_Z()
            if not input.Z.isempty():
                #current_c = new_Z.get_c()
                current_c = new_Z.c
                current_c[index] = sign
                #new_Z.set_c(current_c)
                new_Z.c = current_c
            
            S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z)
            
            result.append(S1)
            
        return S
    
    @staticmethod    
    def stepReach_multiple_inputs(*args):
        """
            Performs reachability analysis on a set of inputs
            
            input : Star* -> a set of input Star-s
            index : int* -> indices where stepReach is to be performed
            mode : string* -> sign function types
            option
            
            returns reachable sets for the given inputs
        """
        
        results = []
        
        input = args[SIGN_STEPREACH_MULT_ARGS_INPUTS_ID]
        
        for i in range(len(input)):
            results.append(self.stepReach(input[i], index[i], mode[i]))
            
        return results
    
    @staticmethod
    def reach_star_exact(*args):
        """
            Performs exact reachability analysis for the given Star
            
            input : Star (or ImageStar) -> input set
            mode : string -. sign function type
            option
            
            returns exact reachable set for the given input
        """
        
        input = args[SIGN_REACH_STAR_EXACT_INPUT_ID]
        mode = args[SIGN_REACH_STAR_EXACT_MODE_ID]
        
        if not input.isEmptySet():
            In = Star()
            
            if isinstance(input, ImageStar):
                In = input.to_star()
            else:
                In.deep_copy(input)
        
        if Sign.isempty(In.predicate_lb) or Sign.isempty(In.predicate_ub):
            new_pred_lb, new_pred_ub = In.getPredicateBounds()
            In.predicate_lb = new_pred_lb
            In.predicate_ub = new_pred_ub
            
        lb, ub = In.estimateRanges()
        
        if Sign.isempty(lb) or Sign.isempty(ub):
            return []
        else:
            for i in range(len(lb)):
                if isinstance(In, Star) or isinstance(In, ImageStar):
                    In = Sign.stepReach(In, i, mode)
                else:
                    In = Sign.stepReach_multiple_inputs(In[i], i, mode[i])
                    
        return In
    
    @staticmethod
    def reach_star_exact_multiple_inputs(*args):
        """
            Performs exact reachability analysis for a set inputs
            
            input : Star* (ImageStar*) -> set of inputs
            mode : string* -> sign function types 
            option
        
            returns exact reachable sets for the given inputs
        """
        
        if isinstance(args[SIGN_RMI_EXACT_ARGS_INPUT_ID], Star) or isinstance(args[SIGN_RMI_EXACT_ARGS_INPUT_ID], ImageStar):
            return Sign.reach_star_exact(args[SIGN_RMI_EXACT_ARGS_INPUT_ID], args[SIGN_RMI_EXACT_ARGS_MODE_ID])
        else:
            result = [] 
            
            for i in range(len(input)):
                result.append(self.reach_star_exact(args[SIGN_RMI_EXACT_ARGS_INPUT_ID][i], self.attributes[SIGN_RMI_EXACT_ARGS_MODE_ID]))
                
            return result
      
    @staticmethod  
    def step_reach_star_approx(input, mode):
        """
            Performs an overapproximate stepReach operation
            
            input : Star -> input Star
            mode : string -> sign function type
            
            returns overapproximate reachable set for the given input
        """
        
        lb, ub = input.estimateRanges()
        
        sign_lb, sign_ub = Sign.get_sign_bounds(mode)
        
        if Sign.isempty(lb) or Sign.isempty(ub):
            return []
        else:
            #current_nVar = input.get_nVar()
            current_nVar = input.nVar
                        
            l = len(lb)
            
            #new_V = input.get_V()
            #new_Z = input.get_Z()
            new_V = input.V
            new_Z = input.Z
                        
            neg_ids = np.argwhere(ub < 0)
            pos_ids = np.argwhere(lb >= 0)
            
            ids = np.transpose(np.array([i for i in range(len(ub))]))
            
            others = np.setdiff1d(ids, neg_ids)
            others = np.setdiff1d(others, pos_ids)
            additional = len(others)
            
            new_V = np.hstack((new_V, np.zeros((l, additional))))
            
            new_V[:, 1 : (current_nVar + 1)] = np.zeros((l, current_nVar))
            new_V[others, current_nVar + 1 : new_V.shape[1]] = np.eye((additional))
            
            
            new_V[neg_ids, 0] = sign_lb
            new_V[pos_ids, 0] = sign_ub
            new_V[others, 0] = Sign.evaluate(new_V[others, 0], mode)
            
            new_C = np.zeros(((2 * additional), current_nVar + additional))
            new_d = np.zeros((additional * 2, 1))
            
            pred_ids = np.transpose(np.array([i for i in range(additional)]))
            
            rows = np.append((pred_ids + 1) * 2 - 2, (pred_ids + 1) * 2 - 1)
            #rows = np.vstack((pred_ids * 2 - 1, pred_ids * 2))
            cols = np.append(current_nVar + pred_ids, current_nVar + pred_ids)
            
            values = np.append(np.zeros((additional, 1)) - 1, np.zeros((additional, 1)) + 1)
            
            new_C[rows, cols] = values
            
            new_d[(pred_ids + 1) * 2 - 2] = 1
            new_d[(pred_ids + 1) * 2 - 1] = 1
            
            #new_pred_lb = np.vstack((input.get_pred_lb, np.zeros((additional, 1))))
            #new_pred_ub = np.vstack((input.get_pred_ub, np.zeros((additional, 1))))
            new_pred_lb = np.append(input.predicate_lb, np.zeros((additional, 1)))
            new_pred_ub = np.append(input.predicate_ub, np.zeros((additional, 1)))
            
            new_pred_lb[pred_ids + current_nVar] = sign_lb
            new_pred_ub[pred_ids + current_nVar] = sign_ub
            
            #new_d = np.vstack((input.get_d(), new_d))
            #new_C = np.vstack((np.hstack((input.get_C(), np.zeros((input.get_C().shape[0], additional)))), new_C))
            new_d = np.vstack((np.reshape(input.d, (input.d.shape[0], 1)), new_d))
            new_C = np.vstack((np.hstack((input.C, np.zeros((input.C.shape[0], additional)))), new_C))
            
            if new_Z.isempty():
                new_Z.c = new_V[:, 0]
                new_Z.V = new_V[:, 1 : new_V.shape[1]]
                
            new_d = np.reshape(new_d, (new_d.shape[0],))
                
            return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_Z)
    
    @staticmethod    
    def reach_star_approx_multiple_inputs(*args):
        """
            Preforms overapproximate reachability analysis on the given input
            input : Star (ImageStar) -> the input set
            mode : string -> sign function type
            
            returns the overapproximate rechable set for the given input
        """
        
        input = args[SIGN_REACH_STAR_APPROX_INPUT_ID]
        mode = args[SIGN_REACH_STAR_APPROX_MODE_ID]
        
        if input.isEmptySet():
            return []
        else:
            In = Star()
            
            imgs_convert = False
            if isinstance(input, ImageStar):
                In = input.to_star()
                imgs_convert = True
            else:
                In.deep_copy(input)
                
            RS = Sign.step_reach_star_approx(In, mode)
                
            if imgs_convert:
                return RS.toImageStar(input.get_height(), input.get_width(), input.get_num_channel())
            else:
                return RS
             
    @staticmethod   
    def reach(*args):
        """
            Performs reachability analysis for the given input
            
            input : Star*(ImageStar*) -> a set of inputs
            method : string* -> reachability methods
            modes : string* -> a set of sign function types
            option
            
            returns reachable sets for the given inputs
        """
        
        #TODO: multiple modes
            
        if args[SIGN_REACH_ARGS_METHOD_ID] == 'exact-star':
            return Sign.reach_star_exact_multiple_inputs(args[SIGN_REACH_ARGS_INPUT_IMAGES_ID], args[SIGN_REACH_ARGS_MODE_ID])
        else:
            return Sign.reach_star_approx_multiple_inputs(args[SIGN_REACH_ARGS_INPUT_IMAGES_ID], args[SIGN_REACH_ARGS_MODE_ID])
            
    @staticmethod
    def isempty(param):
        return param.size == 0 or (param is np.array and param.shape[0] == 0)
    
    @staticmethod
    def get_sign_bounds(mode):
        return Sign.sign_bounds[mode]