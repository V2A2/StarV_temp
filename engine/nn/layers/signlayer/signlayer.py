import numpy as np
import torch
import torch.nn as nn 
import sys

SIGNL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
SIGNL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
SIGNL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

SIGN_POLAR_ZERO_POS_ONE = 'polar_zero_to_pos_one'
SIGN_NONNEGATIVE_ZERO_POS_ONE = 'nonnegative_zero_to_pos_one'

SINGL_DEFAULT_NAME = 'sign_layer'
SINGL_DEFAULT_MODE = SIGN_POLAR_ZERO_POS_ONE
SIGNL_DEFAULT_NUM_INPUTS = 1
SIGNL_DEFAULT_INPUT_NAMES = 2
SIGNL_DEFAULT_NUM_OUTPUTS = 3
SIGNL_DEFAULT_OUTPUT_NAMES = 4

SINGL_DEFAULT_DISPLAY_OPTION = []

sys.path.insert(0, "engine/nn/funcs/sign")
from sign import *

sys.path.insert(0, "engine/set/star")
from star import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *


class SignLayer:
    """
        The Sign layer class in CNN
        Contains constructor and reachability analysis methods    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        if len(args) <= 6 and len(args) > 0:
            assert isinstance(args[0], str), 'error: %s' % SIGNL_ERRMSG_NAME_NOT_STRING 
            self.name = args[0]
            
            if len(args) == 6:
                self.num_inputs = args[2]
                self.input_names = args[3]
                self.num_outputs = args[4]
                self.output_names = args[5]
            else:
                self.num_inputs = SIGNL_DEFAULT_NUM_INPUTS
                self.input_names = SIGNL_DEFAULT_INPUT_NAMES
                self.num_outputs = SIGNL_DEFAULT_NUM_OUTPUTS
                self.output_names = SIGNL_DEFAULT_OUTPUT_NAMES
                
            if len(args) == 6 or len(args) == 2:
                self.attributes[SIGNL_MODE_ID] = args[1]
            else:
                self.attributes[SIGNL_MODE_ID] = SINGL_DEFAULT_MODE
        elif len(args) == 0:
            self.name = SINGL_DEFAULT_NAME
            self.attributes[SIGNL_MODE_ID] = SINGL_DEFAULT_MODE  
            self.num_inputs = SIGNL_DEFAULT_NUM_INPUTS
            self.input_names = SIGNL_DEFAULT_INPUT_NAMES
            self.num_outputs = SIGNL_DEFAULT_NUM_OUTPUTS
            self.output_names = SIGNL_DEFAULT_OUTPUT_NAMES          
        else:
            raise Exception(SIGNL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
            
        return Sign.evaluate(torch.FloatTensor(input), self.attributes[SIGNL_MODE_ID])
            
        #return np.reshape(Sign.evaluate(torch.reshape(torch.FloatTensor(input),(np.prod(input.shape), 1)), self.attributes[SIGNL_MODE_ID]), input.shape)
    
    def reach_star_single_input(self, input, method, option = SINGL_DEFAULT_DISPLAY_OPTION):
        """
            Performs reachability analysis on the given input
                 
            input : ImageStar -> the input ImageStar
            method : string -> reachability method
            option
                
            returns a set of reachable sets for the given input images
        """
             
        assert isinstance(input, ImageStar) or isinstance(input, Star), 'error: %s' % SIGNL_ERRORMSG_INVALID_INPUT
            
        input_image = input
        if isinstance(input, ImageStar):
            input_image = input.to_star()
            
        reachable_sets = Sign.reach(input_image, method, self.attributes[SIGNL_MODE_ID], option)

        if isinstance(input, ImageStar):
            rs = []
            
            for i in range(len(reachable_sets)):
                rs.append(reachable_sets[i].to_image_star(h, w, c))
            
            reachable_sets = rs
            
        return reachable_sets
    
    def reach_star_multiple_inputs(self, input_images, method, option = SINGL_DEFAULT_DISPLAY_OPTION):
        """
            Performs reachability analysis on the given multiple inputs
            
            input_images : ImageStar* -> a set of input images
            method : string -> 'exact-star' - exact reachability analysis
                               'approx-star' - overapproximate reachability analysis
            option
            
            returns an array of reachable sets for the given input images
        """
    
        r_images = []
        
        for i in range(len(input_images)):
            r_images.append(self.reach_star_single_input(input_images[i], method, option))
            
        return r_images
            
    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
            method : string -> 'exact-star' - exact star reachability
                            -> 'approx-star' - approx star reachability
                            -> 'approx-zono' - approx zono reachability
            option : int -> 0 - single core
                         -> 1 - multiple cores
                         
            
            returns the output set(s)
        """
        
        if args[1] == 'approx-star' or args[1] == 'exact-star':
            IS = self.reach_star_multiple_inputs(args[0], args[1])
        else:
            raise Exception(SIGNL_ERRMSG_UNK_REACH_METHOD)
            
        return IS
