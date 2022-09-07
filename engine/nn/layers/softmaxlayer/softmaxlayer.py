import numpy as np
import torch
import torch.nn as nn
import os, sys

SOFTMAXL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
SOFTMAXL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
SOFTMAXL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

SOFTMAXL_DEFAULT_NAME = 'softmax_layer'

SINGL_DEFAULT_DISPLAY_OPTION = []

sys.path.insert(0, "engine/set/star")
from star import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

class SoftmaxLayer:
    """
        The Softmax layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(5):
            self.attributes.append(np.array([]))
        
        if len(args) == 5:
            self.name = self.attributes[0]
            self.num_inputs = self.attributes[1]
            self.input_names = self.attributes[2]
            self.num_outputs = self.attributes[3]
            self.output_names = self.attributes[4]
        elif len(args) == 1:
            assert isinstance(self.attributes[0], str), 'error: %s' % SOFTMAXL_ERRMSG_NAME_NOT_STRING

            self.name = self.attributes[0]
        elif len(args) == 0:
            self.name = SOFTMAXL_DEFAULT_NAME
        else:
            raise Exception(SOFTMAXL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
        
        if isinstance(input, np.ndarray):
            input = torch.FloatTensor(input)
            
        softmax_activation = torch.nn.Softmax(dim=0)
            
        return softmax_activation(input).cpu().detach().numpy()
    
    def reach(self, input):
        """
            input : ImageStar -> imageStar input set
        """
        
        return input
