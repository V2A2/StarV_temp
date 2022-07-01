import numpy as np
import sys, os

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *

sys.path.insert(0, "engine/set/star")
from imagestar import *

sys.path.insert(0, "engine/set/zono")
from imagezono import *


FC_ERRMSG_NAME_NOT_STRING = "Layer name is not a string"
FC_ERRMSG_INPUT_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays" 
FC_ERRMSG_WEGHT_BIAS_INCONSISTENT = "Inconsistent dimension between the weight matrix and bias vector"
FC_ERRMSG_BIAS_SHAPE_INCONSISTENT = "Bias vector should only have one column"
FC_ERRMSG_INVALID_NUMBER_OF_INPUTS = "Invalid number of inputs (should be 0, 2 or 3)"
FC_ERRMSG_INPUT_LAYER_SIZE_INCONSISTENT = "Inconsistency between the input dimension and InputSize of the network"
FC_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"

FCLAYER_ATTRIBUTES_NUM = 6

FC_FULL_ARGS_LEN = 3
FC_CALC_ARGS_LEN = 2
FC_EMPTY_ARGS_LEN = 0

FC_CALC_ARGS_OFFSET = 1

FC_DEFAULT_NAME = "fully_connected_layer"
FC_DEFAULT_EMPTY_NAME = ""
FC_DEFAULT_EMPTY_WEIGHTS = np.array([])
FC_DEFAULT_EMPTY_BIAS = np.array([])
FC_DEFAULT_EMPTY_INPUT_SIZE = 0
FC_DEFAULT_EMPTY_OUTPUT_SIZE = 0

FC_NAME_ID = 0
FC_W_ID = 1
FC_B_ID = 2
FC_INPUT_SIZE_ID = 3
FC_OUTPUT_SIZE_ID = 4
FLATTEN_ORDER_ID = 5

FC_REACH_ARGS_INPUT_IMAGES_ID = 0

COLUMN_FLATTEN = 'F'

class FullyConnectedLayer:
    """
     The FullyConnectedLayer layer class in CNN
     Contain constructor, evaluation, and reachability analysis methods
     Main references:
     1) An intuitive explanation of convolutional neural networks: 
        https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
     2) More detail about mathematical background of CNN
        http://cs231n.github.io/convolutional-networks/
        http://cs231n.github.io/convolutional-networks/pool
     3) Matlab implementation of Convolution2DLayer and MaxPooling (for training and evaluating purpose)
        https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
        https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.fullyconnectedlayer.html
    """

    # Author: Michael Ivashchenko
    
    def __init__(self, *args):        
        self.attributes = []       
        
        for i in range(FCLAYER_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        
        if len(args) == FC_EMPTY_ARGS_LEN:            
            self.attributes[FC_NAME_ID] = FC_DEFAULT_EMPTY_NAME
            
            self.attributes[FC_W_ID] = FC_DEFAULT_EMPTY_WEIGHTS 
            self.attributes[FC_B_ID] = FC_DEFAULT_EMPTY_BIAS
            
            self.attributes[FC_INPUT_SIZE_ID] = FC_DEFAULT_EMPTY_INPUT_SIZE
            self.attributes[FC_OUTPUT_SIZE_ID] = FC_DEFAULT_EMPTY_OUTPUT_SIZE
        elif len(args) == FC_FULL_ARGS_LEN or len(args) == FC_CALC_ARGS_LEN:
            if len(args) == FC_CALC_ARGS_LEN:
                args = self.offset_args(args, FC_CALC_ARGS_OFFSET)
                self.attributes[FC_NAME_ID] = FC_DEFAULT_NAME
            else:
                assert isinstance(args[FC_NAME_ID], str), 'error: %s' % FC_ERRMSG_NAME_NOT_STRING
                self.attributes[FC_NAME_ID] = args[FC_NAME_ID] 
            
            assert isinstance(args[FC_W_ID], np.ndarray) and isinstance(args[FC_B_ID], np.ndarray), 'error: %s' % FC_ERRMSG_WEGHT_OR_BIAS_NOT_NP
            assert args[FC_W_ID].shape[0] == args[FC_B_ID].shape[0], 'error: %s' % FC_ERRMSG_WEGHT_BIAS_INCONSISTENT
            assert (len(args[FC_B_ID].shape) == 2 and args[FC_B_ID].shape[1] == 1) or len(args[FC_B_ID].shape) == 1, 'error: %s' % FC_ERRMSG_BIAS_SHAPE_INCONSISTENT
            
            self.attributes[FC_W_ID] = args[FC_W_ID].astype('float64') 
            self.attributes[FC_B_ID] = args[FC_B_ID].astype('float64')
            
            self.attributes[FC_INPUT_SIZE_ID] = args[FC_W_ID].shape[1]
            self.attributes[FC_OUTPUT_SIZE_ID] = args[FC_W_ID].shape[0]
            
            self.attributes[FLATTEN_ORDER_ID] = COLUMN_FLATTEN
        else:
            raise Exception(FC_ERRMSG_INVALID_NUMBER_OF_INPUTS)

    def evaluate(self, input):
        """
            Evaluates the layer using the given input
            
            input : np.array([*]) -> the input to the layer
            
            returns the result of the input multiplied by the weight matrix and added to the bias
        """        
    
        assert np.prod(input.shape) == self.attributes[FC_INPUT_SIZE_ID], 'error: %s' % FC_ERRMSG_INPUT_LAYER_SIZE_INCONSISTENT
        
        # TODO: create a separate 'is_column_vec' utility function + use it in line 79
        if (len(input.shape) == 2 and input.shape[1] != 1) and len(input.shape) != 1:
            I = input.flatten(order=self.attributes[FLATTEN_ORDER_ID])
        else:
            I = input
            
        y = np.matmul(self.attributes[FC_W_ID], I) + self.attributes[FC_B_ID]
        
        # TODO: consider y = reshape(y, [1, 1, size(obj.Bias, 1)]); when implementing reachability algorithms
        
        return y
    
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis
            
            in_image : ImageStar or ImageZono -> the input image
            
            returns the affine mapping of the input set
        """
        
        assert isinstance(input_image, ImageStar) or \
               isinstance(input_image, ImageZono), 'error: %s' % FC_ERRMSG_INPUT_NOT_IMAGESTAR
        
        input_size = input_image.get_height() * input_image.get_width() * input_image.get_num_channel() 
    
        assert input_size == self.attributes[FC_INPUT_SIZE_ID], 'error: %s' % FC_ERRMSG_INPUT_LAYER_SIZE_INCONSISTENT
        
        input_pred_num = input_image.get_num_pred()
        
        new_V = np.zeros((1, 1, self.attributes[FC_OUTPUT_SIZE_ID], input_pred_num + 1))
        
        for i in range(input_size + 1):
            I = input_image.get_V()[:,:,:,i]
            
            I = I.flatten(order=self.attributes[FLATTEN_ORDER_ID])
        
            if i == 0:
                new_V[0,0,:,i] = np.matmul(self.attributes[FC_W_ID], I) + self.attributes[FC_B_ID]
            else:
                new_V[0,0,:,i] = np.matmul(self.attributes[FC_W_ID], I)
                
        if isinstance(input_image, ImageStar):
            return ImageStar(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        elif isinstance(input_image, ImageZono):
            return ImageZono(new_V, input_image.get_C(), input_image.get_d(), input_image.get_pred_lb(), input_image.get_pred_ub())
        
    def reach_multiple_inputs(self, input_images, option = []):
        """
            Performs reachability analysis on the given set of images
            
            inputs : [Image*) -> the array of ImageStars or ImageZonos
            option : int -> 0 - single core
                         -> 1 - multiple cores
                         
            returns the affine mappings of the input images
        """
        
        output_images = []
        
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
        
        #assert args[FC_ARGS_METHODID] < 5, 'error: %s' % FC_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[FC_REACH_ARGS_INPUT_IMAGES_ID], [])
    
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
