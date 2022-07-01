import numpy as np
import sys
import os

sys.path.insert(0, "engine/set/star")
from star import *

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/zono")
from zono import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *


BN_ERRMSG_NAME_NOT_STRING = "Layer name is not a string"
BN_ERRMSG_UNK_KEY = "One of the given keywords does not match any of the attributes"
BN_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"

BNLAYER_ATTRIBUTES_NUM = 11

BN_NAME_ID = 0
BN_TRAINED_VAR_ID = 1
BN_TRAINED_MEAN_ID = 2
BN_SCALE_ID = 3
BN_OFFSET_ID = 4
BN_EPSILON_ID = 5
BN_NUM_CHANNELS_ID = 6
BN_NUM_INPUTS_ID = 7
BN_NUM_OUTPUTS_ID = 8
BN_INPUT_NAMES_ID = 9
BN_OUTPUT_NAMES_ID = 10

BN_DEFAULT_KEYS_IDS = {
        'Name' : 0,
        'TrainedVariance' : 1,
        'TrainedMean' : 2,
        'Scale' : 3,
        'Offset' : 4,
        'Epsilon' : 5,
        'NumChannels' : 6,
        'NumInputs' : 7,
        'InputNames' : 8,
        'NumOutputs' : 9,
        'OutputNames' : 10
    }

BN_DEFAULT_EMPTY_ARGS = {}

BN_REACH_ARGS_NPUT_IMAGES_ID = 0

COLUMN_FLATTEN = 'F'

class BatchNormalizationLayer:
    """
        The Batch Normalization Layer class in CNN
        Contains constructor, evaluation, and reachability analysis methods
    """

    # Author: Michael Ivashchenko
    
    def __init__(self, args): 
        """
            Constructor
            
            args : {1:1} -> a dictionary that has the attribute names as keys, and their respective values
        """       
        self.attributes = []       
        self.key_id_map = BN_DEFAULT_KEYS_IDS
        
        if args == BN_DEFAULT_EMPTY_ARGS:
            self.init_empty_layer()
        else:
            self.validate_keys(args)
            
            for i in range(BNLAYER_ATTRIBUTES_NUM):
                self.attributes.append(np.array([]))
            
            for key in args.keys():
                self.attributes[self.key_id_map[key]] = args[key]
        
        self.fix_attributes()
        
        # TODO: implement default values assignment
            
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            
            input : np.array([*])
            
            returns the normalized output
        """
        input = self.fix_param(input)
        
        if not self.isempty(self.attributes[BN_TRAINED_MEAN_ID]) and not self.isempty(self.attributes[BN_TRAINED_VAR_ID]) and \
           not self.isempty(self.attributes[BN_OFFSET_ID]) and not self.isempty(self.attributes[BN_SCALE_ID]):
            y = input - self.isempty(self.attributes[BN_TRAINED_MEAN_ID])
                                     
            for i in range(self.attributes[BN_NUM_CHANNELS_ID]):
                y[i,:,:] = np.divide(y[i,:,:], np.sqrt(self.attributes[BN_TRAINED_MEAN_ID] + self.attributes[BN_EPSILON_ID]))
                y[i,:,:] = np.multiply(y[i,:,:], self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
        elif not self.isempty(self.attributes[BN_SCALE_ID]) and not self.isempty(self.attributes[BN_OFFSET_ID]) \
             and self.isempty(self.attributes[BN_TRAINED_MEAN_ID]) and self.isempty(self.attributes[BN_TRAINED_VAR_ID]):
            y = input - self.isempty(self.attributes[BN_TRAINED_MEAN_ID])
            
            if self.attributes[BN_NUM_CHANNELS_ID] == 1:
                y = np.multiply(y, self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
            else:
                for i in range(self.attributes[BN_NUM_CHANNELS_ID]):
                    y[i,:,:] = np.multiply(y[i,:,:], self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
        else:
            y = input
            
        return y
    
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the normalized image
        """
        
        assert isinstance(input_image, Star) or isinstance(input_image, ImageStar)     \
               or isinstance(input_image, Zono) or isinstance(input_image, ImageZono), \
               'error: %s' % BN_ERRORMSG_INVALID_INPUT
               
        output = input_image
               
        if isinstance(input_image, ImageStar) or isinstance(input_image, ImageZono):
            self.fix_param(input_image)
            
            buffer = np.zeros((1, 1, self.attributes[BN_NUM_CHANNEL_ID]))
            
            if not self.isempty(self.attributes[BN_TRAINED_MEAN_ID]) and not self.isempty(self.attributes[BN_TRAINED_VAR_ID]) and \
               not self.isempty(self.attributes[BN_OFFSET_ID]) and not self.isempty(self.attributes[BN_SCALE_ID]):
                for i in range(self.attributes[BN_NUM_CHANNEL_ID]):
                    buffer = 1 / np.sqrt(self.attributes[BN_TRAINED_VAR_ID][0, 0, i] + self.attributes[BN_EPSILON_ID])
                    
                    output = output.affine_map(buffer, np.multiply(-buffer, self.attributes[BN_TRAINED_MEAN_ID]))
                
            output = output.affine_map(self.attributes[BN_SCALE_ID], self.attributes[BN_OFFSET_ID])
            
        else:
            buffer = np.divide(self.attributes[BN_SCALE_ID], np.sqrt(self.attributes[BN_TRAINED_VAR_ID] + self.attributes[BN_EPSILON_ID]))
            
            #TODO: make a getter
            new_V = output.V
            for i in range(new_V.shape[1]):
                new_V[:, i] = (np.multiply(output.V[:,i] - self.attributes[BN_TRAINED_MEAN_ID], buffer) + self.attributes[BN_OFFSET_ID])
                
            #TODO: make a setter
            output.V = new_V
            
        return output
    
    def reach_multiple_inputs(self, input_images, option = []):
        """
            Performs reachability analysis on several input images
            
            input_images : [Image*] -> a set of ImageStar-s or ImageZono-s (Star-s or Zono-s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns a set of normalized images
        """
        
        output_images = []
        
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
        
        assert args[BN_ARGS_METHODID] < 5, 'error: %s' % BN_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[BN_REACH_ARGS_NPUT_IMAGES_ID], [])    
    
########################## UTILS ##########################
    def fix_param(self, param):
        if(len(param.shape) < 3):
            new_shape = np.append([1 for i in range(3 - len(param.shape))], param.shape)
            
            return np.reshape(param, new_shape)
        
        return param

    def fix_attributes(self):
        self.attributes[BN_TRAINED_MEAN_ID] = self.fix_param(self.attributes[BN_TRAINED_MEAN_ID])
        self.attributes[BN_TRAINED_VAR_ID] = self.fix_param(self.attributes[BN_TRAINED_VAR_ID])
        self.attributes[BN_SCALE_ID] = self.fix_param(self.attributes[BN_SCALE_ID])
        self.attributes[BN_OFFSET_ID] = self.fix_param(self.attributes[BN_OFFSET_ID])    

    def init_empty_layer(self):
        for key in self.key_id_map.keys:
            if(key == 'Name'):
                self.attributes[attributes_ids_dict[key]] = ''
            else:
                self.attributes[attributes_ids_dict[key]] = 0
                
    def validate_keys(self, args):
        for key in args.keys():
            assert key in self.key_id_map.keys(), 'error: %s' % BN_ERRMSG_UNK_KEY
            
            if key == 'Name':
                assert isinstance(args[key], str), 'error: %s' % BN_ERRMSG_NAME_NOT_STRING

    def report_missing_param(self, param_name):
        return f'error: Parameter {param_name} missing'
    

    def isempty(self, param):
        return param.size == 0 or (param is np.array and param.shape[0] == 0)

    
        