import numpy as np
import torch
import torch.nn as nn 

CONV2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
CONV2D_ERRMSG_INVALID_PADDING = 'Invalid padding matrix'
CONV2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
CONV2D_ERRMSG_INVALID_DILATION = 'Invalid dilation matrix'
CONV2D_ERRMSG_INVALID_WEIGHTS_SHAPE = 'Invalid weights array'
CONV2D_ERRMSG_INVALID_BIAS_SHAPE = 'Invalid bias array'
CONV2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT = 'Inconsistency between filter weights and filter biases'
CONV2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 2, 3, 5, 6, or 10)'
CONV2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
CONV2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
CONV2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'
CONV2D_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
CONV2D_ERRORMSG_INVALID_INPUT = 'The input should be ImageStar or ImageZono'

CONV2D_ATTRIBUTES_NUM = 13
CONV2D_FULL_ARGS_LEN = 10
CONV2D_FULL_CALC_ARGS_LEN = 6
CONV2D_CALC_ARGS_LEN = 5
CONV2D_FULL_WEIGHTS_BIAS_ARGS_LEN = 3
CONV2D_WEIGHTS_BIAS_ARGS_LEN = 2

CONV2D_NAME_ID = 0
CONV2D_WEIGHTS_ID = 1
CONV2D_BIAS_ID = 2
CONV2D_NUM_FILTERS_ID = 3
CONV2D_NUM_CHANNELS_ID = 4
CONV2D_KERNEL_SIZE_ID = 5
CONV2D_PADDING_ID = 6
CONV2D_STRIDE_ID = 7
CONV2D_DILATION_ID = 8
CONV2D_NUMINPUTS_ID = 9
CONV2D_NUMOUTPUTS_ID = 10
CONV2D_INPUTNAMES_ID = 11
CONV2D_OUTPUTNAMES_ID = 12

CONV2D_ARGS_NAME_ID = 0
CONV2D_ARGS_WEIGHTS_ID = 1
CONV2D_ARGS_BIAS_ID = 2
CONV2D_ARGS_PADDING_ID = 3
CONV2D_ARGS_STRIDE_ID = 4
CONV2D_ARGS_DILATION_ID = 5
CONV2D_ARGS_NUMINPUTS_ID = 6
CONV2D_ARGS_NUMOUTPUTS_ID = 7
CONV2D_ARGS_INPUTNAMES_ID = 8
CONV2D_ARGS_OUTPUTNAMES_ID = 9

CONV2D_EVAL_FULL_ARGS_LEN = 2
CONV2D_EVAL_ARGS_LEN = 1

CONV2D_REACH_ARGS_INPUT_IMAGES_ID = 0
CONV2D_EVAL_ARGS_INPUT_ID = 0

CONV2D_CALC_ARGS_OFFSET = 1

CONV2D_DEFAULT_LAYER_NAME = 'convolutional_layer'
CONV2D_DEFAULT_PADDING = np.array([0,0,0,0])
CONV2D_DEFAULT_STRIDE = np.array([1,1])
CONV2D_DEFAULT_DILATION = np.array([1,1])

class Conv2DLayer:
    """
        The convolutional 2D layer class
        Contain constructor, evaluation, and reachability analysis methods
        Main references:
        1) An intuitive explanation of convolutional neural networks: 
           https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        2) More detail about mathematical background of CNN
           http://cs231n.github.io/convolutional-networks/
        3) Matlab implementation of Convolution2DLayer (for training and evaluating purpose)
           https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.convolution2dlayer.html
    """
    
    def Conv2DLayer(self, *args):
        self.attributes = []       
        
        for i in range(CONV2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) <= CONV2D_FULL_ARGS_LEN:
            if len(args == CONV2D_FULL_ARGS_LEN):
                self.attributes[CONV2D_NUMINPUTS_ID] = args[CONV2D_ARGS_NUMINPUTS_ID]
                self.attributes[CONV2D_NUMOUTPUTS_ID] = args[CONV2D_ARGS_NUMOUTPUTS_ID]
                self.attributes[CONV2D_INPUTNAMES_ID] = args[CONV2D_ARGS_INPUTNAMES_ID]
                self.attributes[CONV2D_OUTPUTNAMES_ID] = args[CONV2D_ARGS_OUTPUTNAMES_ID]
            elif len(args) == CONV2D_FULL_CALC_ARGS_LEN or len(args) == CONV2D_FULL_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONV2D_ARGS_NAME_ID], str), 'error: %s' % CONV2D_ERRMSG_NAME_NOT_STRING
                self.attributes[CONV2D_NAME_ID] = args[CONV2D_ARGS_NAME_ID]

            if len(args) == CONV2D_CALC_ARGS_LEN or len(args) == CONV2D_WEIGHTS_BIAS_ARGS_LEN:
                args = self.offset_args(args, CONV2D_CALC_ARGS_OFFSET)
            
            if len(args) > CONV2D_WEIGHTS_BIAS_ARGS_LEN and len(args) <= CONV2D_CALC_ARGS_LEN:
                self.attributes[CONV2D_NAME_ID] = CONV2D_DEFAULT_LAYER_NAME
                
            if len(args) > CONV2D_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONV2D_ARGS_PADDING_ID], np.ndarray), 'error: %s' % CONV2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONV2D_ARGS_PADDING_ID].shape) == 1 or \
                        len(args[CONV2D_ARGS_PADDING_ID].shape) == 2 or \
                        len(args[CONV2D_ARGS_PADDING_ID].shape) == 4, \
                        'error: %s' % CONV2D_ERRMSG_INVALID_PADDING
                
                assert isinstance(args[CONV2D_ARGS_STRIDE_ID], np.ndarray), 'error: %s' % CONV2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONV2D_ARGS_STRIDE_ID].shape) == 1 or \
                        len(args[CONV2D_ARGS_STRIDE_ID].shape) == 2 or \
                        'error: %s' % CONV2D_ERRMSG_INVALID_STRIDE
                        
                assert isinstance(args[CONV2D_ARGS_DILATION_ID], np.ndarray), 'error: %s' % CONV2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONV2D_ARGS_DILATION_ID].shape) == 1 or \
                        len(args[CONV2D_ARGS_DILATIONE_ID].shape) == 2 or \
                        'error: %s' % CONV2D_ERRMSG_INVALID_DILATION
                        
                self.attributes[CONV2D_PADDING_ID] = args[CONV2D_ARGS_PADDING_ID]
                self.attributes[CONV2D_STRIDE_ID] = args[CONV2D_ARGS_STRIDE_ID]
                self.attributes[CONV2D_DILATION_ID] = args[CONV2D_ARGS_DILATION_ID]
                
            elif len(args) == CONV2D_WEIGHTS_BIAS_ARGS_LEN:
                self.attributes[CONV2D_PADDING_ID] = CONV2D_DEFAULT_PADDING
                self.attributes[CONV2D_STRIDE_ID] =  CONV2D_DEFAULT_STRIDE
                self.attributes[CONV2D_DILATION_ID] = CONV2D_DEFAULT_DILATION
                
                
            assert isinstance(args[CONV2D_ARGS_WEIGHTS_ID], np.ndarray), 'error: %s' % CONV2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONV2D_ARGS_WEIGHTS_ID].shape) > 1, 'error: %s' % CONV2D_ERRMSG_INVALID_WEIGHTS_SHAPE
                
            assert isinstance(args[CONV2D_ARGS_BIAS_ID], np.ndarray), 'error: %s' % CONV2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONV2D_ARGS_BIAS_ID].shape) == 3, 'error: %s' % CONV2D_ERRMSG_INVALID_BIAS_SHAPE
            
            self.attributes[CONV2D_WEIGHTS_ID] = args[CONV2D_ARGS_WEIGHTS_ID]
            self.attributes[CONV2D_BIAS_ID] = args[CONV2D_ARGS_BIAS_ID]
            
            
            if len(args[CONV2D_ARGS_WEIGHTS_ID].shape) == 4:
                assert args[CONV2D_ARGS_WEIGHTS_ID].shape[3] == args[CONV2D_ARGS_BIAS_ID].shape[2], 'error: %s' % CONV2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT
                
                self.attributes[CONV2D_NUM_FILTERS_ID] = self.attributes[CONV2D_WEIGHTS_ID].shape[3]
                self.attributes[CONV2D_NUM_CHANNELS_ID] = self.attributes[CONV2D_WEIGHTS_ID].shape[2]
            elif len(args[CONV2D_ARGS_WEIGHTS_ID].shape) == 2:
                self.attributes[CONV2D_NUM_FILTERS_ID] = 1
                self.attributes[CONV2D_NUM_CHANNELS_ID] = 1 
            elif len(args[CONV2D_ARGS_WEIGHTS_ID].shape) == 3:
                self.attributes[CONV2D_NUM_FILTERS_ID] = 1
                self.attributes[CONV2D_NUM_CHANNELS_ID] = self.attributes[CONV2D_WEIGHTS_ID].shape[2]
            
            self.attributes[CONV2D_KERNEL_SIZE_ID] = np.array([self.attributes[CONV2D_WEIGHTS_ID].shape[0], self.attributes[CONV2D_WEIGHTS_ID].shape[1]])
         
        else:       
            raise Exception(CONV2D_ERRMSG_INVALID_NUMBER_OF_INPUTS)
                
    def evaluate(self, *args):
        """
            Applies convolution2D operation using pytorch functionality
            
            input : np.array([*]) -> a 3D array
            NOT IMPLEMENTED YET => option : str -> 'single' - single precision of computation
                                   'double' - double precision of computation
                                   'empty' - empty precision of computation
                            
            returns a convolved output
        """
        
        current_option = 'double'
        
        if len(args) == CONV2D_EVAL_FULL_ARGS_LEN:
            assert args[CONV2D_EVAL_PRECISION_OPT_ID] == 'single' or \
                   args[CONV2D_EVAL_PRECISION_OPT_ID] == 'double' or \
                   args[CONV2D_EVAL_PRECISION_OPT_ID] == 'empty', \
                   'error: %s' % CONV2D_ERRMSG_INVALID_PRECISION_OPT
        elif len(args) != CONV2D_EVAL_ARGS_LEN:
            raise(CONV2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        conv = nn.Conv2D(in_channels=self.attributes[CONV2D_NUMINPUTS_ID], \
                             out_channels=self.attributes[CONV2D_NUMOUTPUTS_ID], \
                             kernel_size=self.attributes[CONV2D_KERNEL_SIZE_ID].shape, \
                             stride=self.attributes[CONV2D_STRIDE_ID].shape, \
                             padding=self.attributes[CONV2D_PADDING_ID].shape, \
                             dilation=self.attributes[CONV2D_DILATION_ID].shape)
        
            
        conv.weight = torch.nn.Parameter(self.attributes[CONV2D_WEIGHTS_ID])
        conv.bias = torch.nn.Parameter(self.attributes[CONV2D_BIAS_ID])
            
        return conv(args[CONV2D_EVAL_ARGS_INPUT_ID]).cpu().detach().numpy()
       
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the convolved image
        """
        
        assert isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono), \
               'error: %s' % CONV2D_ERRORMSG_INVALID_INPUT
        assert input_image.get_num_channel() == self.attributes[NUM_CHANNELS_ID], \
               'error: %s' % CONV2D_ERRORMSG_INCONSISTENT_CHANNELS_NUM
               
        conv = nn.Conv2D(in_channels=self.attributes[CONV2D_NUMINPUTS_ID], \
                             out_channels=self.attributes[CONV2D_NUMOUTPUTS_ID], \
                             kernel_size=self.attributes[CONV2D_KERNEL_SIZE_ID].shape, \
                             stride=self.attributes[CONV2D_STRIDE_ID].shape, \
                             padding=self.attributes[CONV2D_PADDING_ID].shape, \
                             dilation=self.attributes[CONV2D_DILATION_ID].shape)
        
        conv.weight = torch.nn.Parameter(self.attributes[CONV2D_WEIGHTS_ID])
        conv.bias = torch.nn.Parameter(self.attributes[CONV2D_BIAS_ID])
            
        new_c = conv(input_image.get_V()[:, :, :, 0]).cpu().detach().numpy()
        conv.bias = torch.empty()
        
        new_V = conv(input_image.get_V()[:, :, :, 1 : input_image.get_pred_num() + 2]).cpu().detach().numpy()

        if isinstance(input_image, ImageStar):
            return ImageStar(np.vstack(new_c, new_V), input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        else:
            return ImageZono(np.vstack(new_c, new_V), input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        
    def reach_multiple_inputs(self, input_images, options):
        """
            Performs reachability analysis on several input images
            
            input_images : [Image*] -> a set of ImageStar-s or ImageZono-s (Star-s or Zono-s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns a set of convolved images
        """
        
        output_images = []
        
        if option > 0:
            raise NotImplementedError
        
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
        
        assert args[CONV2D_ARGS_METHODID] < 5, 'error: %s' % CONV2D_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[CONV2D_REACH_ARGS_INPUT_IMAGES_ID], option)    
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
