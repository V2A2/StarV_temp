import numpy as np
import torch
import torch.nn as nn 

CONVTR2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
CONVTR2D_ERRMSG_INVALID_PADDING = 'Invalid padding matrix'
CONVTR2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
CONVTR2D_ERRMSG_INVALID_DILATION = 'Invalid dilation matrix'
CONVTR2D_ERRMSG_INVALID_WEIGHTS_SHAPE = 'Invalid weights array'
CONVTR2D_ERRMSG_INVALID_BIAS_SHAPE = 'Invalid bias array'
CONVTR2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT = 'Inconsistency between filter weights and filter biases'
CONVTR2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 2, 3, 4, 5, or 9)'
CONVTR2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
CONVTR2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
CONVTR2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'

CONVTR2D_ATTRIBUTES_NUM = 12
CONVTR2D_FULL_ARGS_LEN = 9
CONVTR2D_FULL_CALC_ARGS_LEN = 5
CONVTR2D_CALC_ARGS_LEN = 4
CONVTR2D_FULL_WEIGHTS_BIAS_ARGS_LEN = 3
CONVTR2D_WEIGHTS_BIAS_ARGS_LEN = 2

CONVTR2D_NAME_ID = 0
CONVTR2D_WEIGHTS_ID = 1
CONVTR2D_BIAS_ID = 2
CONVTR2D_CROPPING_SIZE_ID = 3
CONVTR2D_NUM_FILTERS_ID = 4
CONVTR2D_NUM_CHANNELS_ID = 5
CONVTR2D_KERNEL_SIZE_ID = 6
CONVTR2D_STRIDE_ID = 7
CONVTR2D_NUMINPUTS_ID = 8
CONVTR2D_NUMOUTPUTS_ID = 9
CONVTR2D_INPUTNAMES_ID = 10
CONVTR2D_OUTPUTNAMES_ID = 11

CONVTR2D_ARGS_NAME_ID = 0
CONVTR2D_ARGS_WEIGHTS_ID = 1
CONVTR2D_ARGS_BIAS_ID = 2
CONVTR2D_ARGS_CROPPING_SIZE_ID = 3
CONVTR2D_ARGS_STRIDE_ID = 4
CONVTR2D_ARGS_NUMINPUTS_ID = 5
CONVTR2D_ARGS_NUMOUTPUTS_ID = 6
CONVTR2D_ARGS_INPUTNAMES_ID = 7
CONVTR2D_ARGS_OUTPUTNAMES_ID = 8

CONVTR2D_EVAL_FULL_ARGS_LEN = 2
CONVTR2D_EVAL_ARGS_LEN = 1

CONVTR2D_REACH_ARGS_INPUT_IMAGES_ID = 0
CONVTR2D_EVAL_ARGS_INPUT_ID = 0

CONVTR2D_CALC_ARGS_OFFSET = 1

CONVTR2D_DEFAULT_LAYER_NAME = 'transposed_convolutional_layer'
CONVTR2D_DEFAULT_CROPPING_SIZE = np.array([0,0,0,0])
CONVTR2D_DEFAULT_STRIDE = np.array([1,1])

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
    
    def ConvTranspose2DLayer(self, *args):
        self.attributes = []       
        
        for i in range(CONVTR2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) <= CONVTR2D_FULL_ARGS_LEN:
            if len(args == CONVTR2D_FULL_ARGS_LEN):
                self.attributes[CONVTR2D_NUMINPUTS_ID] = args[CONVTR2D_ARGS_NUMINPUTS_ID]
                self.attributes[CONVTR2D_NUMOUTPUTS_ID] = args[CONVTR2D_ARGS_NUMOUTPUTS_ID]
                self.attributes[CONVTR2D_INPUTNAMES_ID] = args[CONVTR2D_ARGS_INPUTNAMES_ID]
                self.attributes[CONVTR2D_OUTPUTNAMES_ID] = args[CONVTR2D_ARGS_OUTPUTNAMES_ID]
            elif len(args) == CONVTR2D_FULL_CALC_ARGS_LEN or len(args) == CONVTR2D_FULL_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONVTR2D_ARGS_NAME_ID], str), 'error: %s' % CONVTR2D_ERRMSG_NAME_NOT_STRING
                self.attributes[CONVTR2D_NAME_ID] = args[CONVTR2D_ARGS_NAME_ID]

            if len(args) == CONVTR2D_CALC_ARGS_LEN or len(args) == CONVTR2D_WEIGHTS_BIAS_ARGS_LEN:
                args = self.offset_args(args, CONVTR2D_CALC_ARGS_OFFSET)
            
            if len(args) > CONVTR2D_WEIGHTS_BIAS_ARGS_LEN and len(args) <= CONVTR2D_CALC_ARGS_LEN:
                self.attributes[CONVTR2D_NAME_ID] = CONVTR2D_DEFAULT_LAYER_NAME
                
            if len(args) > CONVTR2D_WEIGHTS_BIAS_ARGS_LEN:
                assert isinstance(args[CONVTR2D_ARGS_STRIDE_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONVTR2D_ARGS_STRIDE_ID].shape) == 1 or \
                        len(args[CONVTR2D_ARGS_STRIDE_ID].shape) == 2 or \
                        'error: %s' % CONVTR2D_ERRMSG_INVALID_STRIDE
                        
                assert isinstance(args[CONVTR2D_ARGS_CROPPING_SIZE_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
                assert len(args[CONVTR2D_ARGS_CROPPING_SIZE_ID].shape) == 1 or \
                        len(args[CONVTR2D_ARGS_CROPPING_SIZEE_ID].shape) == 2 or \
                        'error: %s' % CONVTR2D_ERRMSG_INVALID_CROPPING_SIZE
                        
                self.attributes[CONVTR2D_STRIDE_ID] = args[CONVTR2D_ARGS_STRIDE_ID]
                self.attributes[CONVTR2D_CROPPING_SIZE_ID] = args[CONVTR2D_ARGS_CROPPING_SIZE_ID]
                
            elif len(args) == CONVTR2D_WEIGHTS_BIAS_ARGS_LEN:
                self.attributes[CONVTR2D_STRIDE_ID] =  CONVTR2D_DEFAULT_STRIDE
                self.attributes[CONVTR2D_CROPPING_SIZE_ID] = CONVTR2D_DEFAULT_CROPPING_SIZE
                
                
            assert isinstance(args[CONVTR2D_ARGS_WEIGHTS_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 4, 'error: %s' % CONVTR2D_ERRMSG_INVALID_WEIGHTS_SHAPE
                
            assert isinstance(args[CONVTR2D_ARGS_BIAS_ID], np.ndarray), 'error: %s' % CONVTR2D_ERRMSG_PARAM_NOT_NP
            assert len(args[CONVTR2D_ARGS_BIAS_ID].shape) == 3, 'error: %s' % CONVTR2D_ERRMSG_INVALID_BIAS_SHAPE
            
            if len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 4:
                assert args[CONVTR2D_ARGS_WEIGHTS_ID].shape[2] == args[CONVTR2D_ARGS_BIAS_ID].shape[2], 'error: %s' % CONVTR2D_ERRMSG_WEIGHTS_BIAS_INCONSISTENT
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[3]
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[2]
            
                self.attributes[CONVTR2D_KERNEL_SIZE_ID] = np.array([self.attributes[CONVTR2D_WEIGHTS_ID].shape[0], self.attributes[CONVTR2D_WEIGHTS_ID].shape[1]])
            elif len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 2:
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = 1
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = 1 
            elif len(args[CONVTR2D_ARGS_WEIGHTS_ID].shape) == 3:
                self.attributes[CONVTR2D_NUM_FILTERS_ID] = 1
                self.attributes[CONVTR2D_NUM_CHANNELS_ID] = self.attributes[CONVTR2D_WEIGHTS_ID].shape[2]
                
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
        
        if len(args) == CONVTR2D_EVAL_FULL_ARGS_LEN:
            assert args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'single' or \
                   args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'double' or \
                   args[CONVTR2D_EVAL_PRECISION_OPT_ID] == 'empty', \
                   'error: %s' % CONVTR2D_ERRMSG_INVALID_PRECISION_OPT
        elif len(args) != CONVTR2D_EVAL_ARGS_LEN:
            raise(CONVTR2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        conv = nn.ConvTranspose2d(in_channels=self.attributes[CONVTR2D_NUMINPUTS_ID], \
                                  out_channels=self.attributes[CONVTR2D_NUMOUTPUTS_ID], \
                                  kernel_size=self.attributes[CONVTR2D_KERNEL_SIZE_ID].shape, \
                                  stride=self.attributes[CONVTR2D_STRIDE_ID].shape, \
                                  padding=self.attributes[CONVTR2D_CROPPING_SIZE_ID].shape)
        
            
        conv.weight = torch.nn.Parameter(self.attributes[CONVTR2D_WEIGHTS_ID])
        conv.bias = torch.nn.Parameter(self.attributes[CONVTR2D_BIAS_ID])
            
        return conv(args[CONVTR2D_EVAL_ARGS_INPUT_ID]).cpu().detach().numpy()
       
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the convolved image
        """
        
        assert isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono), \
               'error: %s' % CONVTR2D_ERRORMSG_INVALID_INPUT
        assert input_image.get_num_channel() == self.attributes[NUM_CHANNELS_ID], \
               'error: %s' % CONVTR2D_ERRORMSG_INCONSISTENT_CHANNELS_NUM
               
        conv = nn.ConvTranspose2d(in_channels=self.attributes[CONVTR2D_NUMINPUTS_ID], \
                                  out_channels=self.attributes[CONVTR2D_NUMOUTPUTS_ID], \
                                  kernel_size=self.attributes[CONVTR2D_KERNEL_SIZE_ID].shape, \
                                  stride=self.attributes[CONVTR2D_STRIDE_ID].shape, \
                                  padding=self.attributes[CONVTR2D_CROPPING_SIZE_ID].shape)
        
        conv.weight = torch.nn.Parameter(self.attributes[CONVTR2D_WEIGHTS_ID])
        conv.bias = torch.nn.Parameter(self.attributes[CONVTR2D_BIAS_ID])
            
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
        
        assert args[CONVTR2D_ARGS_METHODID] < 5, 'error: %s' % CONVTR2D_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[CONVTR2D_REACH_ARGS_INPUT_IMAGES_ID], option)    
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
