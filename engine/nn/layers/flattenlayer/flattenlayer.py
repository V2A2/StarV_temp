import numpy as np
import torch
import torch.nn as nn 
import sys

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *

FLATL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
FLATL_ERRORMSG_INVALID_INPUT_IMG = 'Invalid input image'
FLATL_ERRORMSG_UNK_FLATL_TYPE = 'Unknown type of flatten layer'
FLATL_ERRORMSG_INVALID_INPUT = 'The input should be ImageStar or ImageZono'

7 = 7

7 = 7
3 = 3
2 = 2
0 = 0

FLATL_NAME_ID = 0
FLATL_MODE_ID = 1
FLATL_TYPE_ID = 2
FLATL_NUM_INPUTS_ID = 3
FLATL_INPUT_NAMES_ID = 4
FLATL_NUM_OUTPUTS_ID = 5
FLATL_OUTPUT_NAMES_ID = 6

0 = 0
1 = 1
2 = 2
3 = 3
4 = 4
5 = 5
6 = 6

0 = 0

FLATL_CALC_ARGS_OFFSET = 1

COLUMN_FLATTEN = 'F'
FLATL_CSTYLE_TYPE = 'nnet.keras.layer.FlattenCStyleLayer'
FLATL_NNET_TYPE = 'nnet.cnn.layer.FlattenLayer'
FLATL_DEFAULT_DISPLAY_OPTION = []

FLATL_DEFAULT_NAME = 'flatten_layer'
FLATL_DEFAULT_MODE = COLUMN_FLATTEN
FLATL_DEFAULT_TYPE = 'nnet.keras.layer.FlattenCStyleLayer'
FLATL_DEFAULT_NUM_INPUTS = 1
FLATL_DEFAULT_NUM_OUTPUTS = 1
FLATL_DEFAULT_INPUT_NAMES = ['in']
FLATL_DEFAULT_OUTPUT_NAMES = ['out']

class FlattenLayer:
    """
        Flatten Layer object
    """
    
    def __init__(self, *args):
        """
            name : string -> name of the layer
            num_inputs : int -> number of inputs
            num_outputs : int -> number of outputs
            input_names : string* -> input_names
            output_names : string* -> output_names
        """
        
        if len(args) <= 7:
            if len(args) == 7:
                assert args[3] > 0, 'error: %s' % FLATL_ERRMSG_INVALID_INPUTS_NUM_ID
                self.num_inputs = args[3]
                
                assert args[5] > 0, 'error: %s' % FLATL_ERRMSG_INVALID_OUTPUTS_NUM_ID
                self.num_outputs = args[5]
                
                self.input_names = args[4]
                self.output_names = args[6]
            else:
                self.num_inputs = FLATL_DEFAULT_NUM_INPUTS
                self.num_outputs = FLATL_DEFAULT_NUM_OUTPUTS
                self.input_names = FLATL_DEFAULT_INPUT_NAMES
                self.output_names = FLATL_DEFAULT_OUTPUT_NAMES

            if len(args) > 2 :
                assert isinstance(args[0], str), 'error: %s' % FLATL_ERRMSG_NAME_NOT_STRING
                self.name = args[0]
            else:
                self.name = FLATL_DEFAULT_NAME
                                
            if len(args) > 0:
                self.mode = args[1]
                self.type = args[2]
            else:
                self.mode = FLATL_DEFAULT_MODE
                self.type = FLATL_DEFAULT_TYPE

    def evaluate(self, input):
        """
            Evaluates the layer using the given image
            
            input : np.array([*]) -> a multi-channel image
            
            returns a flattened image
        """
        
        input_size = input.shape
        input = torch.FloatTensor(input)
        flatten_image = []
        
        if self.type == FLATL_CSTYLE_TYPE:
            if len(input_size) == 2:
                flatten_im = torch.permute(input, (1,0))
                flatten_im = torch.reshape(flatten_im, (1, np.prod(input_size)))
            elif len(input_size) == 3:
                if self.mode == COLUMN_FLATTEN:
                    flatten_im = torch.permute(input, (2,1,0))
                else:
                    flatten_im = torch.permute(input, (2,0,1))
                
                flatten_im = torch.reshape(flatten_im, (1, 1, np.prod(input_size)))
            else:
                raise Exception(FLATL_ERRORMSG_INVALID_INPUT_IMG)
        elif self.type == FLATL_NNET_TYPE:
            if len(input_size) == 2:
                flatten_im = torch.reshape(input, (1, np.prod(input_size)))
            elif len(input_size) == 3:
                flatten_im = torch.reshape(input, (1, 1, np.prod(input_size)))
            else:
                raise Exception(FLATL_ERRORMSG_INVALID_INPUT_IMG)
        else:
            raise Exception(FLATL_ERRORMSG_UNK_FLATL_TYPE)
        
        return flatten_im.cpu().detach().numpy()
        
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : Image -> input ImageStar or ImageZono
            
            returns the reachable set of the given image
        """    
        
        assert isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono), \
               'error: %s' % FLATL_ERRORMSG_INVALID_INPUT
        
        total_input_size = input_image.get_height() * input_image.get_width() * input_image.get_num_channel()
        
        input_num_pred = input_image.get_num_pred()
        
        new_V = np.zeros((1,1,total_input_size, input_num_pred + 1))
        
        input_V = input_image.get_V()
        
        for i in range(input_num_pred + 1):
            new_V[0,0,:,i] = self.evaluate(input_V[:,:,:,i])
            
        if isinstance(input_image, ImageStar):
            return ImageStar(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        else:
            return ImageZono(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())


    def reach_multiple_inputs(self, input_images, option = FLATL_DEFAULT_DISPLAY_OPTION):
        """
            Performs reachability analysis on a single input image
            
            input_image : Image* -> input ImageStar-s or ImageZono-s
            
            returns the reachable sets of the given images
        """
        
        result = []
        for i in range(len(input_images)):
            result.append(self.reach_single_input(input_images[i]))
            
        return result
    
    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
                         
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns the output set(s)
        """
                
        IS = self.reach_multiple_inputs(args[0])    

########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result