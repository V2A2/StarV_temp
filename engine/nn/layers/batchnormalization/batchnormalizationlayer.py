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

        self.key_id_map = {
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
        
        if args == {}:
            self.init_empty_layer()
        else:
            self.validate_keys(args)
            
            for i in range(11):
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
        
        if not self.isempty(self.trained_mean) and not self.isempty(self.trained_var) and \
           not self.isempty(self.offset) and not self.isempty(self.scale):
            y = input - self.isempty(self.trained_mean)
                                     
            for i in range(self.num_channel):
                y[i,:,:] = np.divide(y[i,:,:], np.sqrt(self.trained_mean + self.epsilon))
                y[i,:,:] = np.multiply(y[i,:,:], self.scale) + self.offset
        elif not self.isempty(self.scale) and not self.isempty(self.offset) \
             and self.isempty(self.trained_mean) and self.isempty(self.trained_var):
            y = input - self.isempty(self.trained_mean)
            
            if self.num_channel == 1:
                y = np.multiply(y, self.scale) + self.offset
            else:
                for i in range(self.num_channel):
                    y[i,:,:] = np.multiply(y[i,:,:], self.scale) + self.offset
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
            
            buffer = np.zeros((1, 1, self.num_channel))
            
            if not self.isempty(self.trained_mean) and not self.isempty(self.trained_var) and \
               not self.isempty(self.offset) and not self.isempty(self.scale):
                for i in range(self.num_channel):
                    buffer = 1 / np.sqrt(self.trained_var[0, 0, i] + self.epsilon)
                    
                    output = output.affine_map(buffer, np.multiply(-buffer, self.trained_mean))
                
            output = output.affine_map(self.scale, self.offset)
            
        else:
            buffer = np.divide(self.scale, np.sqrt(self.trained_var + self.epsilon))
            
            #TODO: make a getter
            new_V = output.V
            for i in range(new_V.shape[1]):
                new_V[:, i] = (np.multiply(output.V[:,i] - self.trained_mean, buffer) + self.offset)
                
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
        
        assert args[1] < 5, 'error: %s' % BN_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[0], [])    
    
########################## UTILS ##########################
    def fix_param(self, param):
        if(len(param.shape) < 3):
            new_shape = np.append([1 for i in range(3 - len(param.shape))], param.shape)
            
            return np.reshape(param, new_shape)
        
        return param

    def fix_attributes(self):
        self.trained_mean = self.fix_param(self.trained_mean)
        self.trained_var = self.fix_param(self.trained_var)
        self.scale = self.fix_param(self.scale)
        self.offset = self.fix_param(self.offset)    

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

    
        