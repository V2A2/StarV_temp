import numpy as np
import torch
import torch.nn as nn 
import sys

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *


AVGP2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
AVGP2D_ERRMSG_INVALID_POOL_SIZE = 'Invalid pool size'
AVGP2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
AVGP2D_ERRMSG_INVALID_PADDING_SIZE = 'Invalide padding size'
AVGP2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 3, or 4)'
AVGP2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
AVGP2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
AVGP2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'
AVGP2D_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
AVGP2DERRORMSG_INVALID_INPUT = 'The given input image is not an ImageStar or ImageZono'

AVGP2D_ATTRIBUTES_NUM = 8

AVGP2D_FULL_ARGS_LEN = 8
AVGP2D_FULL_CALC_ARGS_LEN = 4
AVGP2D_CALC_ARGS_LEN = 3

AVGP2D_NAME_ID = 0
AVGP2D_POOL_SIZE_ID = 1
AVGP2D_STRIDE_ID = 2
AVGP2D_PADDING_SIZE_ID = 3
AVGP2D_NUM_INPUTS_ID = 4
AVGP2D_INPUT_NAMES_ID = 5
AVGP2D_NUM_OUTPUTS_ID = 6
AVGP2D_OUTPUT_NAMES_ID = 7

AVGP2D_ARGS_NAME_ID = 0
AVGP2D_ARGS_POOL_SIZE_ID = 1
AVGP2D_ARGS_STRIDE_ID = 2
AVGP2D_ARGS_PADDING_SIZE_ID = 3
AVGP2D_ARGS_NUM_INPUTS_ID = 4
AVGP2D_ARGS_INPUT_NAMES_ID = 5
AVGP2D_ARGS_NUM_OUTPUTS_ID = 6
AVGP2D_ARGS_OUTPUT_NAMES_ID = 7

AVGP2D_CALC_ARGS_OFFSET = 1

AVGP2D_EVAL_ARGS_INPUT_ID = 0

AVGP2D_EVAL_FULL_ARGS_LEN = 2
AVGP2D_EVAL_ARGS_LEN = 1

AVGP2D_REACH_ARGS_INPUT_IMAGES_ID = 0

AVGP2D_DEFAULT_LAYER_NAME = 'average_pooling_layer'
AVGP2D_DEFAULT_POOL_SIZE = (2, 2)
AVGP2D_DEFAULT_STRIDE = (1, 1)
AVGP2D_DEFAULT_PADDING_SIZE = (0,0)

class AveragePooling2DLayer:
    """
        The average pooling 2D layer class
        Contains constructor, evaluation, and reachability analysis methods
        
        Main references:
        1) An intuitive explanation of convolutional neural networks: 
           https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        2) More detail about mathematical background of CNN
           http://cs231n.github.io/convolutional-networks/
        3) Matlab implementation of Convolution2DLayer and AveragePooling2dLayer (for training and evaluating purpose)
           https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.convolution2dlayer.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.averagepooling2dlayer.html
    """
    
    def __init__(self, *args):
        self.attributes = []       
        
        for i in range(AVGP2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) <= AVGP2D_FULL_ARGS_LEN and len(args) > 0:
            if len(args)== AVGP2D_FULL_ARGS_LEN or len(args) == AVGP2D_FULL_CALC_ARGS_LEN:
                if len(args)== AVGP2D_FULL_ARGS_LEN:
                    self.attributes[AVGP2D_NUM_INPUTS_ID] = args[AVGP2D_ARGS_NUM_INPUTS_ID]
                    self.attributes[AVGP2D_NUM_OUTPUTS_ID] = args[AVGP2D_ARGS_NUM_OUTPUTS_ID]
                    self.attributes[AVGP2D_INPUT_NAMES_ID] = args[AVGP2D_ARGS_INPUT_NAMES_ID]
                    self.attributes[AVGP2D_OUTPUT_NAMES_ID] = args[AVGP2D_ARGS_OUTPUT_NAMES_ID]
                
                assert isinstance(args[AVGP2D_ARGS_NAME_ID], str), 'error: %s' % AVGP2D_ERRMSG_NAME_NOT_STRING
                self.attributes[AVGP2D_NAME_ID] = args[AVGP2D_ARGS_NAME_ID]

            if len(args) == AVGP2D_CALC_ARGS_LEN:
                args = self.offset_args(args, AVGP2D_CALC_ARGS_OFFSET)
                self.attributes[AVGP2D_NAME_ID] = AVGP2D_DEFAULT_LAYER_NAME                
                
            if self.isempty(args[AVGP2D_ARGS_POOL_SIZE_ID]):
                self.attributes[AVGP2D_POOL_SIZE_ID] = AVGP2D_DEFAULT_POOL_SIZE
                self.attributes[AVGP2D_STRIDE_ID] =  AVGP2D_DEFAULT_STRIDE
                self.attributes[AVGP2D_PADDING_SIZE_ID] = AVGP2D_DEFAULT_PADDING_SIZE
            else:
                
                assert isinstance(args[AVGP2D_ARGS_POOL_SIZE_ID], np.ndarray), 'error: %s' % AVGP2D_ERRMSG_PARAM_NOT_NP
                assert args[AVGP2D_ARGS_POOL_SIZE_ID].shape[0] == 1 and \
                       args[AVGP2D_ARGS_POOL_SIZE_ID].shape[1] == 2, \
                        'error: %s' % AVGP2D_ERRMSG_INVALID_POOL_SIZE
                    
                assert isinstance(args[AVGP2D_ARGS_STRIDE_ID], np.ndarray), 'error: %s' % AVGP2D_ERRMSG_PARAM_NOT_NP
                assert args[AVGP2D_ARGS_STRIDE_ID].shape[0] == 1 and \
                       args[AVGP2D_ARGS_STRIDE_ID].shape[1] == 2, \
                        'error: %s' % AVGP2D_ERRMSG_INVALID_STRIDE
                            
                assert isinstance(args[AVGP2D_ARGS_PADDING_SIZE_ID], np.ndarray), 'error: %s' % AVGP2D_ERRMSG_PARAM_NOT_NP
                assert args[AVGP2D_ARGS_PADDING_SIZE_ID].shape[0] == 1 and \
                       args[AVGP2D_ARGS_PADDING_SIZE_ID].shape[1] == 2, \
                        'error: %s' % AVGP2D_ERRMSG_INVALID_PADDING_SIZE
                            
                self.attributes[AVGP2D_POOL_SIZE_ID] = [args[AVGP2D_ARGS_POOL_SIZE_ID].astype('int')[0][i] for i in range(args[AVGP2D_ARGS_POOL_SIZE_ID].shape[1])]
                self.attributes[AVGP2D_STRIDE_ID] = [args[AVGP2D_ARGS_STRIDE_ID].astype('int')[0][i] for i in range(args[AVGP2D_ARGS_STRIDE_ID].shape[1])]
                self.attributes[AVGP2D_PADDING_SIZE_ID] = [args[AVGP2D_PADDING_SIZE_ID].astype('int')[0][i] for i in range(args[AVGP2D_PADDING_SIZE_ID].shape[1])]
        elif len(args) == 0:
                self.attributes[AVGP2D_NAME_ID] = AVGP2D_DEFAULT_LAYER_NAME                

                self.attributes[AVGP2D_POOL_SIZE_ID] = AVGP2D_DEFAULT_POOL_SIZE
                self.attributes[AVGP2D_STRIDE_ID] = AVGP2D_DEFAULT_STRIDE
                self.attributes[AVGP2D_PADDING_SIZE_ID] = AVGP2D_DEFAULT_PADDING_SIZE

        else:       
            raise Exception(AVGP2D_ERRMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, *args):
        """
            Applies averagepooling2D operation using pytorch functionality
            
            input : np.array([*]) -> a 3D array
            NOT IMPLEMENTED YET => option : str -> 'single' - single precision of computation
                                   'double' - double precision of computation
                                   'empty' - empty precision of computation
                            
            returns the pooled output
        """
        
        current_option = 'double'
        
        if len(args) == AVGP2D_EVAL_FULL_ARGS_LEN:
            assert args[AVGP2D_EVAL_PRECISION_OPT_ID] == 'single' or \
                   args[AVGP2D_EVAL_PRECISION_OPT_ID] == 'double' or \
                   args[AVGP2D_EVAL_PRECISION_OPT_ID] == 'empty', \
                   'error: %s' % AVGP2D_ERRMSG_INVALID_PRECISION_OPT
        elif len(args) != AVGP2D_EVAL_ARGS_LEN:
            raise(AVGP2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        avgpool = nn.AvgPool2d(kernel_size=self.attributes[AVGP2D_POOL_SIZE_ID], \
                             stride=self.attributes[AVGP2D_STRIDE_ID], \
                             padding=self.attributes[AVGP2D_PADDING_SIZE_ID])
        
        
        input = args[AVGP2D_EVAL_ARGS_INPUT_ID]
        if not isinstance(input, torch.FloatTensor):
            input = torch.FloatTensor(input)
            
        return avgpool(input).cpu().detach().numpy()
    
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the pooled image
        """

        assert isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono), \
               'error: %s' % AVGP2DERRORMSG_INVALID_INPUT
               
               
        input = torch.permute(torch.FloatTensor(input_image.get_V()), (3,2,0,1))
               
        new_V = torch.permute(torch.FloatTensor(self.evaluate(input)), (2,3,1,0)).cpu().detach().numpy()
        
        if isinstance(input_image, ImageStar):
            return ImageStar(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        else:
            return ImageZono(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())

    def reach_multiple_inputs(self, input_images, options = []):
        """
            Performs reachability analysis on several input images
            
            input_images : [Image*] -> a set of ImageStar-s or ImageZono-s (Star-s or Zono-s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns a set of pooled images
        """
        
        output_images = []
        
        # if option > 0:
        #     raise NotImplementedError
        
        for i in range(len(input_images)):
            output_images.append(self.reach_single_input(input_images[i]))
            
        return output_images

    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
            method : string -> 'exact-star' - exact star reachability
                            -> 'approx-star' - approx star reachability
                            -> 'abs-dom' - abs dom reachability
                            -> 'relax-star' - relax star reachability
                            -> 'approx-zono' - approx zono reachability
                         
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns the output set(s)
        """
        
        # assert args[AVGP2D_ARGS_METHODID] < 5, 'error: %s' % AVGP2DERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[AVGP2D_REACH_ARGS_INPUT_IMAGES_ID])    
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
    
    def isempty(self, param):
        return param.size == 0 or (param is np.array and param.shape[0] == 0)
