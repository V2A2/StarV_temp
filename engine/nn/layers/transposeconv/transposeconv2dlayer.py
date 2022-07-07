import numpy as np
import torch
import torch.nn as nn 
import sys

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *

CONVTR2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
CONVTR2D_ERRMSG_INVALID_PADDING = 'Invalid padding matrix'
CONVTR2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
CONVTR2D_ERRMSG_INVALID_DILATION = 'Invalid dilation matrix'
CONVTR2D_ERRMSG_INVALID_WEIGHTS_SHAPE = 'Invalid weights array'
CONVTR2D_ERRMSG_INVALID_BIAS_SHAPE = 'Invalid bias array'
CONVTR2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT = 'Inconsistency between filter weights and filter biases'
CONVTR2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 2, 3, 5, 6, or 13)'
CONVTR2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
CONVTR2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
CONVTR2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'
CONVTR2D_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'

CONVTR2D_ATTRIBUTES_NUM = 13
CONVTR2D_FULL_ARGS_LEN = 13
CONVTR2D_FULL_CALC_ARGS_LEN = 6
CONVTR2D_CALC_ARGS_LEN = 5
CONVTR2D_FULL_WEIGHTS_BIAS_ARGS_LEN = 3
CONVTR2D_WEIGHTS_BIAS_ARGS_LEN = 2

CONVTR2D_NAME_ID = 0
CONVTR2D_WEIGHTS_ID = 1
CONVTR2D_BIAS_ID = 2
CONVTR2D_PADDING_ID = 3
CONVTR2D_STRIDE_ID = 4
CONVTR2D_DILATION_ID = 5
CONVTR2D_NUMINPUTS_ID = 6
CONVTR2D_INPUTNAMES_ID = 7
CONVTR2D_NUMOUTPUTS_ID = 8
CONVTR2D_OUTPUTNAMES_ID = 9
CONVTR2D_NUM_FILTERS_ID = 10
CONVTR2D_FILTERS_SIZE_ID = 11
CONVTR2D_NUM_CHANNELS_ID = 12

CONVTR2D_ARGS_NAME_ID = 0
CONVTR2D_ARGS_WEIGHTS_ID = 1
CONVTR2D_ARGS_BIAS_ID = 2
CONVTR2D_ARGS_PADDING_ID = 3
CONVTR2D_ARGS_STRIDE_ID = 4
CONVTR2D_ARGS_DILATION_ID = 5
CONVTR2D_ARGS_NUMINPUTS_ID = 6
CONVTR2D_ARGS_INPUTNAMES_ID = 7
CONVTR2D_ARGS_NUMOUTPUTS_ID = 8
CONVTR2D_ARGS_OUTPUTNAMES_ID = 9
CONVTR2D_ARGS_NUM_FILTERS_ID = 10
CONVTR2D_ARGS_FILTERS_SIZE_ID = 11
CONVTR2D_ARGS_NUM_CHANNELS_ID = 12

CONVTR2D_EVAL_FULL_ARGS_LEN = 2
CONVTR2D_EVAL_ARGS_LEN = 1

CONVTR2D_REACH_ARGS_INPUT_IMAGES_ID = 0
CONVTR2D_EVAL_ARGS_INPUT_ID = 0

CONVTR2D_CALC_ARGS_OFFSET = 1

CONVTR2D_DEFAULT_LAYER_NAME = 'transposed_convolutional_layer'
CONVTR2D_DEFAULT_PADDING = np.array([0,0])
CONVTR2D_DEFAULT_STRIDE = np.array([1,1])
CONVTR2D_DEFAULT_DILATION = np.array([1,1])

class ConvTranspose2DLayer:
    """
        The transposed convolutional 2D layer class
        Contains constructor, evaluation, and reachability analysis methods
        Main references:
        1) An intuitive explanation of convolutional neural networks: 
           https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        2) More detail about mathematical background of CNN
           http://cs231n.github.io/convolutional-networks/
        3) Matlab implementation of Convolution2DLayer (for training and evaluating purpose)
           https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.convolution2dlayer.html
    """
    
    def __init__(self, *args):
        self.attributes = []       
        
        for i in range(CONVTR2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) <= CONVTR2D_FULL_ARGS_LEN:
            if len(args) == CONVTR2D_FULL_ARGS_LEN:
                self.attributes[CONVTR2D_NUMINPUTS_ID] = args[CONVTR2D_ARGS_NUMINPUTS_ID].astype('int')
                self.attributes[CONVTR2D_NUMOUTPUTS_ID] = args[CONVTR2D_ARGS_NUMOUTPUTS_ID].astype('int')
                self.attributes[CONVTR2D_INPUTNAMES_ID] = args[CONVTR2D_ARGS_INPUTNAMES_ID]
                self.attributes[CONVTR2D_OUTPUTNAMES_ID] = args[CONVTR2D_ARGS_OUTPUTNAMES_ID]
            elif len(args) == CONVTR2D_FULL_CALC_ARGS_LEN or len(args) == CONVTR2D_FULL_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONVTR2D_ARGS_NAME_ID], str), 'error: %s' % CONVTR2D_ERRMSG_NAME_NOT_STRING
                self.attributes[CONVTR2D_NAME_ID] = args[CONVTR2D_ARGS_NAME_ID]

            if len(args) == CONVTR2D_CALC_ARGS_LEN or len(args) == CONVTR2D_WEIGHTS_BIAS_ARGS_LEN:
                args = self.offset_args(args, CONVTR2D_CALC_ARGS_OFFSET)
                self.attributes[CONVTR2D_NAME_ID] = CONVTR2D_DEFAULT_LAYER_NAME
                
            if len(args) > CONVTR2D_FULL_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONVTR2D_ARGS_PADDING_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONVTR2D_ARGS_PADDING_ID].shape) == 1 or \
                        len(args[CONVTR2D_ARGS_PADDING_ID].shape) == 2 or \
                        len(args[CONVTR2D_ARGS_PADDING_ID].shape) == 4, \
                        'error: %s' % CONVTR2D_ERRMSG_INVALID_PADDING
                
                assert isinstance(args[CONVTR2D_ARGS_STRIDE_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONVTR2D_ARGS_STRIDE_ID].shape) == 1 or \
                        len(args[CONVTR2D_ARGS_STRIDE_ID].shape) == 2 or \
                        'error: %s' % CONVTR2D_ERRMSG_INVALID_STRIDE
                        
                assert isinstance(args[CONVTR2D_ARGS_DILATION_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONVTR2D_ARGS_DILATION_ID].shape) == 1 or \
                        len(args[CONVTR2D_ARGS_DILATIONE_ID].shape) == 2 or \
                        'error: %s' % CONVTR2D_ERRMSG_INVALID_DILATION
                        
                self.attributes[CONVTR2D_PADDING_ID] = args[CONVTR2D_ARGS_PADDING_ID].astype('int')
                self.attributes[CONVTR2D_STRIDE_ID] = args[CONVTR2D_ARGS_STRIDE_ID].astype('int')
                self.attributes[CONVTR2D_DILATION_ID] = args[CONVTR2D_ARGS_DILATION_ID].astype('int')
                
            elif len(args) == CONVTR2D_WEIGHTS_BIAS_ARGS_LEN:
                self.attributes[CONVTR2D_PADDING_ID] = CONVTR2D_DEFAULT_PADDING
                self.attributes[CONVTR2D_STRIDE_ID] =  CONVTR2D_DEFAULT_STRIDE
                self.attributes[CONVTR2D_DILATION_ID] = CONVTR2D_DEFAULT_DILATION
                
                
            assert isinstance(args[CONVTR2D_ARGS_WEIGHTS_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) > 1, 'error: %s' % CONVTR2D_ERRMSG_INVALID_WEIGHTS_SHAPE
                
            assert isinstance(args[CONVTR2D_ARGS_BIAS_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONVTR2D_ARGS_BIAS_ID].shape) == 3, 'error: %s' % CONVTR2D_ERRMSG_INVALID_BIAS_SHAPE
            
            self.attributes[CONVTR2D_WEIGHTS_ID] = args[CONVTR2D_ARGS_WEIGHTS_ID]
            self.attributes[CONVTR2D_BIAS_ID] = args[CONVTR2D_ARGS_BIAS_ID]
            
            
            if len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 4:
                assert args[CONVTR2D_ARGS_WEIGHTS_ID].shape[3] == args[CONVTR2D_ARGS_BIAS_ID].shape[2], 'error: %s' % CONVTR2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT
                
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[3]
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[2]
            elif len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 2:
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = 1
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = 1 
            elif len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 3:
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = 1
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[2]
            
            self.attributes[CONVTR2D_FILTERS_SIZE_ID] = np.array([self.attributes[CONVTR2D_WEIGHTS_ID].shape[0], self.attributes[CONVTR2D_WEIGHTS_ID].shape[1]])
         
        else:       
            raise Exception(CONVTR2D_ERRMSG_INVALID_NUMBER_OF_INPUTS)
                
    def evaluate(self, *args):
        """
            Applies transposed convolution2D operation using pytorch functionality
            
            input : np.array([*]) -> a 3D array
            NOT IMPLEMENTED YET => option : str -> 'single' - single precision of computation
                                   'double' - double precision of computation
                                   'empty' - empty precision of computation
                            
            returns a convolved output
        """
        
        current_option = 'double'
        
        # if len(args) == CONVTR2D_EVAL_FULL_ARGS_LEN:
        #     assert args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'single' or \
        #            args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'double' or \
        #            args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'empty', \
        #            'error: %s' % CONVTR2D_ERRMSG_INVALID_PRECISION_OPT
        # elif len(args) != CONVTR2D_EVAL_ARGS_LEN:
        #     raise(CONVTR2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        convtr = nn.ConvTranspose2d(in_channels=self.attributes[CONVTR2D_NUMINPUTS_ID], \
                                  out_channels=self.attributes[CONVTR2D_NUMOUTPUTS_ID], \
                                  kernel_size=self.attributes[CONVTR2D_FILTERS_SIZE_ID].shape, \
                                  stride=self.attributes[CONVTR2D_STRIDE_ID].shape, \
                                  padding=self.attributes[CONVTR2D_PADDING_ID].shape, \
                                  dilation=self.attributes[CONVTR2D_DILATION_ID].shape)
        
            
        convtr.weight = torch.nn.Parameter(torch.permute(torch.FloatTensor(self.attributes[CONVTR2D_WEIGHTS_ID]), (2,3,0,1)))
        convtr.bias = torch.nn.Parameter(torch.FloatTensor(np.reshape(self.attributes[CONVTR2D_BIAS_ID], (self.attributes[CONVTR2D_BIAS_ID].shape[2],))))
            
        return convtr(torch.FloatTensor(args[CONVTR2D_EVAL_ARGS_INPUT_ID])).cpu().detach().numpy()
       
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the convolved image
        """
        
        assert isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono), \
               'error: %s' % CONVTR2D_ERRORMSG_INVALID_INPUT
        assert input_image.get_num_channel() == self.attributes[CONVTR2D_NUM_CHANNELS_ID], \
               'error: %s' % CONVTR2D_ERRORMSG_INCONSISTENT_CHANNELS_NUM
               
        convtr = nn.ConvTranspose2d(in_channels=self.attributes[CONVTR2D_NUMINPUTS_ID], \
                                  out_channels=self.attributes[CONVTR2D_NUMOUTPUTS_ID], \
                                  kernel_size=self.attributes[CONVTR2D_FILTERS_SIZE_ID].shape, \
                                  stride=self.attributes[CONVTR2D_STRIDE_ID].shape, \
                                  padding=self.attributes[CONVTR2D_PADDING_ID].shape)
        
        convtr.weight = torch.nn.Parameter(torch.permute(torch.FloatTensor(self.attributes[CONVTR2D_WEIGHTS_ID]), (2,3,0,1)))
        convtr.bias = torch.nn.Parameter(torch.FloatTensor(np.zeros((self.attributes[CONVTR2D_NUM_FILTERS_ID],))))
            
        current_V = input_image.get_V()
            
        input_c = np.reshape(current_V[:, :, :, 0], (current_V.shape[0], current_V.shape[1], current_V.shape[2]))
        torch_input_V = torch.permute(torch.FloatTensor(input_c), (2, 0, 1))
        new_c = torch.permute(convtr(torch_input_V), (1, 2, 0)).cpu().detach().numpy()
        
        predicate_V = input_image.get_V()[:, :, :, 1 : input_image.get_num_pred() + 1]
        torch_predicate_V = torch.permute(torch.FloatTensor(predicate_V), (3, 2, 0, 1))
        new_pred_V = torch.permute(convtr(torch_predicate_V), (2, 3, 1, 0)).cpu().detach().numpy()
            
        new_V = np.zeros(np.append(new_c.shape, new_pred_V.shape[3] + 1))
        new_V[:,:,:,0] = new_c
        new_V[:,:,:,1 : new_V.shape[3]] = new_pred_V
            
        if isinstance(input_image, ImageStar):
            return ImageStar(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        else:
            return ImageZono(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())

    def reach_multiple_inputs(self, input_images, options=[]):
        """
            Performs reachability analysis on several input images
            
            input_images : [Image*] -> a set of ImageStar-s or ImageZono-s (Star-s or Zono-s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns a set of convolved images
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
                         
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns the output set(s)
        """
                
        IS = self.reach_multiple_inputs(args[CONVTR2D_REACH_ARGS_INPUT_IMAGES_ID])    
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
