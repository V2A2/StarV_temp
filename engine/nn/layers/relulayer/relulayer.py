import numpy as np
import torch
import torch.nn as nn 
import sys, os

RELUL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
RELUL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
RELUL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

RELUL_DEFAULT_NAME = 'relu_layer'
RELUL_DEFAULT_RELAXFACTOR = 0

sys.path.insert(0, "engine/nn/funcs/poslin")
from poslin import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/star")
from star import *

class ReLULayer:
    """
        The Relu layer class in CNN
        Contains constructor and reachability analysis methods
        Main references:
        1) An intuitive explanation of convolutional neural networks: 
           https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        2) More detail about mathematical background of CNN
           http://cs231n.github.io/convolutional-networks/
           http://cs231n.github.io/convolutional-networks/#pool
        3) Matlab implementation of Convolution2DLayer and MaxPooling (for training and evaluating purpose)
           https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.relulayer.html
    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(5):
            self.attributes.append(np.array([]))
        
        if len(args) == 5:
            self.name = args[0]
            self.num_inputs = args[1]
            self.input_names = args[2]
            self.num_outputs = args[3]
            self.output_names = args[4]
        elif len(args) == 1:
            assert isinstance(args[0], str), 'error: %s' % RELUL_ERRMSG_NAME_NOT_STRING

            self.name = args[0]
        elif len(args) == 0:
            self.name = RELUL_DEFAULT_NAME
        else:
            raise Exception(RELUL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(_, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
            
        return np.reshape(PosLin.evaluate(torch.reshape(torch.FloatTensor(input),(np.prod(input.shape), 1)).cpu().detach().numpy()), input.shape)
    
    def reach_star_single_input(self, input, method, option = [], relax_factor = RELUL_DEFAULT_RELAXFACTOR):
        """
            Performs reachability analysis on the given input
                 
            input : ImageStar -> the input ImageStar
            method : string -> reachability method
            relax_factor : double -> relaxation factor for over-approximate star reachability
                
            returns a set of reachable sets for the given input images
        """
             
        #assert isinstance(input, ImageStar), 'error: %s' % RELUL_ERRORMSG_INVALID_INPUT
            
        input_image = input
            
        if isinstance(input, ImageStar):
            input_image = input_image.to_star()
            
        reachable_sets = PosLin.reach(input_image, method, option, relax_factor)

        
        if isinstance(input, ImageStar):
            rs = []
            for star in reachable_sets:
                rs.append(star.toImageStar(input.get_height(), input.get_width(), input.get_num_channel()))
                
            return rs
            
        return reachable_sets
    
    def reach_star_multiple_inputs(self, input_images, method, option = [], relax_factor = RELUL_DEFAULT_RELAXFACTOR):
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
            r_images.append(self.reach_star_single_input(input_images[i], method, option, relax_factor))
            
        return r_images
        
    def reach_zono(self, input_image):
        """
            Performs reachability analysis on the given input using zonotope
            
            input_image : ImageZono -> the input set
            
            returns a reachable set or the given ImageZono
        """    
        
        assert isinstance(input, ImageZono), 'error: %s' % RELUL_ERRORMSG_INVALID_INPUT
        
        reachable_set = PosLin.reach(input_image.toZono())
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
        
        if args[1] == 'approx-star' or args[1] == 'exact-star':
            IS = self.reach_star_multiple_inputs(args[0], args[1])
        elif args[1] == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[0], args[2])
        else:
            raise Exception(RELUL_ERRMSG_UNK_REACH_METHOD)
            
        return IS
