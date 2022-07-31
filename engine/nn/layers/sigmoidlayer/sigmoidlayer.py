import numpy as np
import torch
import torch.nn as nn 
import sys, os

SIGMOIDL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
SIGMOIDL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
SIGMOIDL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

SIGMOIDL_ATTRIBUTES_NUM = 5

SIGMOIDL_FULL_ARGS_LEN = 5
SIGMOIDL_NAME_ARGS_LEN = 1
SIGMOIDL_EMPTY_ARGS_LEN = 0

SIGMOIDL_NAME_ID = 0
SIGMOIDL_NUM_INPUTS_ID = 1
SIGMOIDL_INPUT_NAMES_ID = 2
SIGMOIDL_NUM_OUTPUTS_ID = 3
SIGMOIDL_OUTPUT_NAMES_ID = 4

SIGMOIDL_ARGS_NAME_ID = 0
SIGMOIDL_ARGS_NUM_INPUTS_ID = 1
SIGMOIDL_ARGS_INPUT_NAMES_ID = 2
SIGMOIDL_ARGS_NUM_OUTPUTS_ID = 3
SIGMOIDL_ARGS_OUTPUT_NAMES_ID = 4

SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID = 0
SIGMOIDL_REACH_ARGS_METHOD_ID = 1
SIGMOIDL_REACH_ARGS_OPTION_ID = 2
SIGMOIDL_REACH_ARGS_RELAX_FACTOR_ID = 3

SIGMOIDL_DEFAULT_NAME = 'sigmoid_layer'
SIGMOIDL_DEFAULT_RELAXFACTOR = 0

sys.path.insert(0, "engine/nn/funcs/logsig")
from logsig import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/star")
from star import *

class SigmoidLayer:
    """
        The Sigmoid layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(SIGMOIDL_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
        
        if len(args) == SIGMOIDL_FULL_ARGS_LEN:
            self.attributes[SIGMOIDL_NAME_ID] = args[SIGMOIDL_ARGS_NAME_ID]
            self.attributes[SIGMOIDL_NUM_INPUTS_ID] = args[SIGMOIDL_ARGS_NUM_INPUTS_ID]
            self.attributes[SIGMOIDL_INPUT_NAMES_ID] = args[SIGMOIDL_ARGS_INPUT_NAMES_ID]
            self.attributes[SIGMOIDL_NUM_OUTPUTS_ID] = args[SIGMOIDL_ARGS_NUM_OUTPUTS_ID]
            self.attributes[SIGMOIDL_OUTPUT_NAMES_ID] = args[SIGMOIDL_ARGS_OUTPUT_NAMES_ID]
        elif len(args) == SIGMOIDL_NAME_ARGS_LEN:
            assert isinstance(args[SIGMOIDL_ARGS_NAME_ID], str), 'error: %s' % SIGMOIDL_ERRMSG_NAME_NOT_STRING

            self.attributes[SIGMOIDL_NAME_ID] = args[SIGMOIDL_ARGS_NAME_ID]
        elif len(args) == SIGMOIDL_EMPTY_ARGS_LEN:
            self.attributes[SIGMOIDL_NAME_ID] = SIGMOIDL_DEFAULT_NAME
        else:
            raise Exception(SIGMOIDL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
            
        return LogSig.evaluate(torch.FloatTensor(input), self.attributes[SIGNL_MODE_ID])
            
        #return np.reshape(LogSig.evaluate(torch.reshape(torch.FloatTensor(input),(np.prod(input.shape), 1)).cpu().detach().numpy()), input.shape)
    
    def reach_star_single_input(self, input, method, option = [], relax_factor = SIGMOIDL_DEFAULT_RELAXFACTOR):
        """
            Performs reachability analysis on the given input
                 
            input : ImageStar -> the input ImageStar
            method : string -> reachability method
            relax_factor : double -> relaxation factor for over-approximate star reachability
                
            returns a set of reachable sets for the given input images
        """
             
        #assert isinstance(input, ImageStar), 'error: %s' % SIGMOIDL_ERRORMSG_INVALID_INPUT
            
        input_image = input
            
        if isinstance(input, ImageStar):
            input_image = input_image.to_star()
            
        reachable_sets = LogSig.reach(input_image, method, option, relax_factor)

        
        if isinstance(input, ImageStar):
            rs = []
            for star in reachable_sets:
                rs.append(star.toImageStar(input.get_height(), input.get_width(), input.get_num_channel()))
                
            return rs
            
        return reachable_sets
    
    def reach_star_multiple_inputs(self, input_images, method, option = [], relax_factor = SIGMOIDL_DEFAULT_RELAXFACTOR):
        """
            Performs reachability analysis on the given multiple inputs
            
            input_images : ImageStar* -> a set of input images
            method : string -> 'exact-star' - exact reachability analysis
                               'approx-star' - overapproximate reachability analysis
            option
            relax_factor : float -> a relaxation factor for overapproximate star reachability
            
            returns an array of reachable sets for the given input images
        """
    
        r_images = []
        
        for i in range(len(input_images)):
            r_images.append(self.reach_star_single_input(input_images[i], method, relax_factor))
            
        return r_images
        
    def reach_zono(self, input_image):
        """
            Performs reachability analysis on the given input using zonotope
            
            input_image : ImageZono -> the input set
            
            returns a reachable set or the given ImageZono
        """    
        
        assert isinstance(input, ImageZono), 'error: %s' % SIGMOIDL_ERRORMSG_INVALID_INPUT
        
        reachable_set = LogSig.reach(input_image.toZono())
        return reachable_set.toImageZono(input_image.get_height(), \
                                         input_image.get_width(), \
                                         input_image.get_channel())
        
    def reach_zono_multiple_inputs(self, input_images, option):
        """
            Performs reachability analysis on the given set of ImageZono-s
            
            input_images : ImageZono* -> a set of images
            
            returns reachable sets of the input images
        """
        
        rs = []
        for i in range(len(input_images)):
            rs.append(self.reach_zono(input_images[i]))

        return rs
    
    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            method : string -> 'exact-star' - exact star reachability
                            -> 'approx-star' - approx star reachability
                            -> 'approx-zono' - approx zono reachability
                         
            
            returns the output set(s)
        """
        
        if args[SIGMOIDL_REACH_ARGS_METHOD_ID] == 'approx-star' or args[SIGMOIDL_REACH_ARGS_METHOD_ID] == 'exact-star':
            IS = self.reach_star_multiple_inputs(args[SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGMOIDL_REACH_ARGS_METHOD_ID])
        elif args[SIGMOIDL_REACH_ARGS_METHOD_ID] == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGMOIDL_REACH_ARGS_OPTION_ID])
        else:
            raise Exception(SIGMOIDL_ERRMSG_UNK_REACH_METHOD)
            
        return IS
