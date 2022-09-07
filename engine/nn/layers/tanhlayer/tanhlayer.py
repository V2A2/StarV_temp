import numpy as np
import torch
import torch.nn as nn 
import sys, os

TANHL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
TANHL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
TANHL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

5 = 5

5 = 5
1 = 1
0 = 0

TANHL_NAME_ID = 0
TANHL_NUM_INPUTS_ID = 1
TANHL_INPUT_NAMES_ID = 2
TANHL_NUM_OUTPUTS_ID = 3
TANHL_OUTPUT_NAMES_ID = 4

0 = 0
1 = 1
2 = 2
3 = 3
4 = 4

0 = 0
1 = 1
2 = 2
3 = 3

TANHL_DEFAULT_NAME = 'tanh_layer'
TANHL_DEFAULT_RELAXFACTOR = 0

sys.path.insert(0, "engine/nn/funcs/tanh")
from tanh import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/star")
from star import *


class TanhLayer:
    """
        The Tanh layer class in CNN
        Contains constructor and reachability analysis methods    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        if len(args) == 5:
            self.name = args[0]
            self.num_inputs = args[1]
            self.input_names = args[2]
            self.num_outputs = args[3]
            self.output_names = args[4]
        elif len(args) == 1:
            assert isinstance(args[0], str), 'error: %s' % TANHL_ERRMSG_NAME_NOT_STRING

            self.name = args[0]
        elif len(args) == 0:
            self.name = TANHL_DEFAULT_NAME
        else:
            raise Exception(TANHL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
         
        return TanSig.evaluate(torch.FloatTensor(input))    
        #return np.reshape(TanSig.evaluate(torch.reshape(input,(np.prod(input.shape), 1))), input.shape)
    
    def reach_star_single_input(_, input, method, relax_factor):
        """
            Performs reachability analysis on the given input
                 
            input : ImageStar -> the input ImageStar
            method : string -> reachability method
            relax_factor : double -> relaxation factor for over-approximate star reachability
                
            returns a set of reachable sets for the given input images
        """
             
        #assert isinstance(input, ImageStar), 'error: %s' % TANHL_ERRORMSG_INVALID_INPUT
         
        input_image = input
            
        if isinstance(input, ImageStar):
            input_image = input_image.to_star()
 
            
        reachable_sets = TanSig.reach(input_image, method, option, relax_factor)

        if isinstance(input, ImageStar):
            rs = []
            for star in reachable_sets:
                rs.append(star.toImageStar(input.get_height(), input.get_width(), input.get_num_channel()))
                
            return rs
            
        return reachable_sets
    
    def reach_star_multiple_inputs(self, input_images, method, option, relax_factor):
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
        
        assert isinstance(input, ImageZono), 'error: %s' % TANHL_ERRORMSG_INVALID_INPUT
        
        reachable_set = TanSig.reach(input_image.toZono())
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
        
        if args[2] == 'approx-star' or args[2] == 'exact-star':
            IS = self.reach_star_multiple_inputs(args[0], args[2], args[1], args[3])
        elif args[2] == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[0], args[1])
        else:
            raise Exception(TANHL_ERRMSG_UNK_REACH_METHOD)
            
        return IS
