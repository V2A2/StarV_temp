import numpy as np
import torch
import torch.nn as nn 
from eagerpy.framework import sign
from tensorflow.python.distribute.device_util import current

SIGN_POLAR_ZERO_POS_ONE = 'polar_zero_to_pos_one'
SIGN_NONNEGATIVE_ZERO_POS_ONE = 'nonnegative_zero_to_pos_one'

class Sign:
    """
        Sign class contains method for reachability analysis for Layer with
        Sign activation function
    """
    
    def evaluate(self, input, mode):
        """
            Evaluates the layer using the given input
            
            input : np.array([*]) -> multi-dimensional array
            mode : string -> sign function type
            
            returns the results of applying Sign to the given input
        """
        
        y = torch.sign(input).cpu().detach().numpy()
        
        if mode == SIGN_POLAR_ZERO_POS_ONE:
            y = y + (y == 0)
        elif mode == SIGN_NONNEGATIVE_ZERO_POS_ONE:
            y = y + (y == 0)
            y = y + (y == -1)
        else:
            raise Exception(SIGN_ERRMSG_MODE_NOT_SUPPORTED)
            
        return y
    
    def stepReach(self, *args):
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
        
        xmin = input.get_min(index)
        xmax = input.get_max(index)
        
        sign_lb = self.get_sign_lb(mode)
        sign_ub = self.get_sign_ub(mode)
        sign = sign_lb
        
        if xmin == xmax and ((amin == sign_lb) or (xmin == sign_ub)):
            return input
        elif xmin >= sign_lb:
            sign = sign + sign_ub - sign_lb
            
        new_V = input.get_V()
        new_V[index, 0] = sign
        
        new_predicate_lb = input.get_pred_lb()
        new_predicate_ub = input.get_pred_ub()
        
        new_predicate_lb[index] = sign
        new_predicate_ub[index] = sign
        new_V[index, 1 : new_V.shape[1]] = np.zeros((1, new_V.shape[2]) - 1)
        
        new_C = input.get_C()
        new_d = input.get_d()
        
        if not self.isempty(input.get_Z()):
            c1 = input.get_Z().get_c()
            c[index] = sign
            V1 = input.get_Z().get_V()
            
            V1[index, 1 : V1.shape[1]] = np.zeros((1, V1.shape[1] - 1))
            
            new_Z = Zono(c1, V1)
        else:
            new_Z = []
            
        S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z)
        
        result.append(S1)
        
        if sign != sign_ub and xmax > 0:
            sign = 1
            new_V[index, 0] = sign
            
            new_predicate_lb[index] = sign
            new_predicate_ub[index] = sign
            
            if not self.isempty(input.get_Z()):
                current_c = new_Z.get_c()
                current_c[index] = sign
                new_Z.set_c(current_c)
            
            S1 = Star(new_V, new_C, new_d, new_predicate_lb, new_predicate_ub, new_Z)
            
            result.append(S1)
            
        return S
        
    def stepReach_multiple_inputs(self, *args):
        """
            Performs reachability analysis on a set of inputs
            
            input : Star* -> a set of input Star-s
            index : int* -> indices where stepReach is to be performed
            option
            mode : string* -> sign function types
            
            returns reachable sets for the given inputs
        """
        
        results = []
        
        input = args[SIGN_STEPREACH_MULT_ARGS_INPUTS_ID]
        
        for i in range(len(input)):
            results.append(self.stepReach(input[i], index[i], mode[i]))
            
        return results
    
    def reach_star_exact(self, *args):
        """
            Performs exact reachability analysis for the given Star
            
            input : Star (or ImageStar) -> input set
            option
            mode : string -. sign function type
            
            returns exact reachable set for the given input
        """
        
        input = args[SIGN_REACH_STAR_EXACT_INPUT_ID]
        mode = args[SIGN_REACH_STAR_EXACT_MODE_ID]
        
        if not input.isempty():
            In = Star()
            
            if isinstance(input, ImageStar):
                In = input.to_star()
            else:
                In.deep_copy(input)
        
        if self.isempty(In.get_pred_lb()) or self.isempty(In.get_pred_ub()):
            new_pred_lb, new_pred_ub = In.get_predicate_bounds()
            In.set_predicate_bounds(new_pred_lb, new_pred_ub)
            
        lb, ub = In.estimate_ranges()
        
        if self.isempty(lb) or self.isempty(ub):
            return []
        else:
            for i in range(len(lb)):
                if(len(In) == 1):
                    In = self.stepReach(In, i, mode)
                else:
                    In = self.stepReach_multiple_inputs(In[i], i, mode[i])
                    
        return In
    
    def reach_star_exact_multiple_inputs(self, *args):
        """
            Performs exact reachability analysis for a set inputs
            
            input : Star* (ImageStar*) -> set of inputs
            option
            mode : string* -> sign function types 
        
            returns exact reachable sets for the given inputs
        """
        
        if len(input) == 1:
            return self.reach_star_exact(input, mode)
        else:
            result = [] 
            
            for i in range(len(input)):
                result.append(self.reach_star_exact(input[i], mode[i]))
                
            return result
        
    def step_reach_star_approx(self, input, mode):
        """
            Performs an overapproximate stepReach operation
            
            input : Star -> input Star
            mode : string -> sign function type
            
            returns overapproximate reachable set for the given input
        """
        
        lb, ub = input.estimateRange()
        
        sign_lb = Sign.get_sign_lb(mode)
        sign_ub = Sign.get_sign_ub(mode)
        
        if self.isempty(lb) or self.isempty(ub):
            return []
        else:
            current_nVar = input.get_nVar()
            
            l = len(lb)
            
            new_V = input.get_V()
            new_Z = input.get_Z()
                        
            neg_ids = np.argwhere(ub < 0)
            pos_ids = np.argwhere(lb >= 0)
            
            ids = np.transpose(np.array([i for i in range(len(ub))]))
            
            others = np.setdiff(ids, neg_ids)
            others = np.setdiff(others, pos_ids)
            additional = len(others)
            
            new_V = np.hstack((new_V, np.zeros((l, additional))))
            
            new_V[:, 1 : (current_nVar + 1)] = np.zeros((l, current_nVar))
            new_V[others, current_nVar + 2 : new_V.shape[1]] = np.eye((additional))
            
            
            new_V[neg_ids, 0] = sign_lb
            new_V[pos_ids, 0] = sign_ub
            new_V[others, 0] = Sign.evaluate(new_V[others, 0]. mode)
            
            new_C = np.zeros(((2 * additional), current_nVar + additional))
            new_d = np.zeros((additional * 2, 1))
            
            pred_ids = np.transpose(np.array([i for i in range(additional)]))
            
            rows = np.vstack((pred_ids * 2 - 1, pred_ids * 2))
            cols = np.vstack((current_nVar + pred_ids, current_nVar + pred_ids))
            
            values = np.vstack((np.zeros((additional, 1)), np.zeros((additional, 1)) + 1))
            
            new_C[np.sub2ind(new_C.shape, rows, cols)] = values
            
            new_d[pred_ids * 2 - 1] = 1
            new_d[pred_ids * 2] = 1
            
            new_pred_lb = np.vstack((In.get_pred_lb, np.zeros((additional, 1))))
            new_pred_ub = np.vstack((In.get_pred_ub, np.zeros((additional, 1))))
            
            new_pred_lb[pred_ids + current] = sign_lb
            new_pred_ub[pred_ids + current] = sign_ub
            
            new_d = np.vstack((In.get_d(), new_d))
            new_C = np.vstack((np.hstack((In.get_C(), np.zeros((input.get_C().shape[0], additional)))), new_C))
            
            if In.get_Z().isempty():
                new_Z.set_c(new_V[:, 0])
                new_Z.set_V(new_V[:, 1 : new_V.shape[1]])
                
            return Star(new_V, new_C, new_d, new_pred_lb, new_pred_ub, new_Z)
        
    def reach_star_approx(self, input, mode):
        """
            Preforms overapproximate reachability analysis on the given input
            input : Star (ImageStar) -> the input set
            mode : string -> sign function type
            
            returns the overapproximate rechable set for the given input
        """
        
        if input.isempty():
            return []
        else:
            In = Star()
            
            imgs_convert = False
            if isinstance(input, ImageStar):
                In = input.to_star()
                imgs_convert = True
            else:
                In.deep_copy(input)
                
            RS = Sign.step_reach_star_approx(self, In, mode)
                
            if imgs_convert:
                return RS.toImageStar(input.get_height(), input.get_width(), input.get_num_channel())
            else:
                return RS
                
    def reach(self, *args):
        """
            Performs reachability analysis for the given input
            
            input : Star*(ImageStar*) -> a set of inputs
            method : string* -> reachability methods
            option
            modes : string* -> a set of sign function types
            
            returns reachable sets for the given inputs
        """
            
        if method == 'exact-star':
            return Sign.reach_star_exact_multiple_inputs(args)
        else:
            return Sign.reach_star_approx_multiple_inputs(args)
            