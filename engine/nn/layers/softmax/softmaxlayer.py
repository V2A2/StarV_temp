import numpy as np
import torch
import torch.nn as nn 

SOFTMAXL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
SOFTMAXL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
SOFTMAXL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

SOFTMAXL_ATTRIBUTES_NUM = 5

SOFTMAXL_FULL_ARGS_LEN = 5
SOFTMAXL_NAME_ARGS_LEN = 1
SOFTMAXL_EMPTY_ARGS_LEN = 0

SOFTMAXL_NAME_ID = 0
SOFTMAXL_NUM_INPUTS_ID = 1
SOFTMAXL_INPUT_NAMES_ID = 2
SOFTMAXL_NUM_OUTPUTS_ID = 3
SOFTMAXL_OUTPUT_NAMES_ID = 4

SOFTMAXL_NAME_ARGS_ID = 0
SOFTMAXL_NUM_INPUTS_ARGS_ID = 1
SOFTMAXL_INPUT_NAMES_ARGS_ID = 2
SOFTMAXL_NUM_OUTPUTS_ARGS_ID = 3
SOFTMAXL_OUTPUT_NAMES_ARGS_ID = 4

SOFTMAXL_REACH_ARGS_INPUT_IMAGES_ID = 0
SOFTMAXL_REACH_ARGS_OPTION_ID = 1
SOFTMAXL_REACH_ARGS_METHOD_ID = 2
SOFTMAXL_REACH_ARGS_RELAX_FACTOR_ID = 3

SOFTMAXL_DEFAULT_NAME = 'relu_layer'

class SoftmaxLayer:
    """
        The Softmax layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def SOFTMAXLayer(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(SOFTMAXL_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
        
        if len(args) == SOFTMAXL_FULL_ARGS_LEN:
            self.attributes[SOFTMAXL_NAME_ID] = self.attributes[SOFTMAXL_ARGS_NAME_ID]
            self.attributes[SOFTMAXL_NUM_INPUTS_ID] = self.attributes[SOFTMAXL_ARGS_NUM_INPUTS_ID]
            self.attributes[SOFTMAXL_INPUT_NAMES_ID] = self.attributes[SOFTMAXL_ARGS_INPUT_NAMES_ID]
            self.attributes[SOFTMAXL_NUM_OUTPUTS_ID] = self.attributes[SOFTMAXL_ARGS_NUM_OUTPUTS_ID]
            self.attributes[SOFTMAXL_OUTPUT_NAMES_ID] = self.attributes[SOFTMAXL_ARGS_OUTPUT_NAMES_ID]
        elif len(args) == SOFTMAXL_NAME_ARGS_LEN:
            assert isinstance(self.attributes[SOFTMAXL_ARGS_NAME_ID], str), 'error: %s' % SOFTMAXL_ERRMSG_NAME_NOT_STRING

            self.attributes[SOFTMAXL_NAME_ID] = self.attributes[SOFTMAXL_ARGS_NAME_ID]
        elif len(args) == SOFTMAXL_EMPTY_ARGS_LEN:
            self.attributes[SOFTMAXL_NAME_ID] = SOFTMAXL_DEFAULT_NAME
        else:
            raise Exception(SOFTMAXL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(_, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
            
        return torch.softmax(input).cpu().detach().numpy()
    
    def reach(self, input):
        """
            input : ImageStar -> imageStar input set
        """
        
        return input
