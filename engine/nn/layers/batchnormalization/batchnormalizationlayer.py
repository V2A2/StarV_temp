import numpy as np

BN_ERRMSG_NAME_NOT_STRING = "Layer name is not a string"
BN_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"

BNLAYER_ATTRIBUTES_NUM = 11

BN_DEFAULT_KEYS_IDS = {
        'Name' : 0,
        'NumChannels' : 1,
        'TrainedMean' : 2,
        'TrainedVariance' : 3,
        'Epsilon' : 4,
        'Offset' : 5,
        'Scale' : 6,
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
            
            for key in self.key_id_map.keys:
                assert key in args.keys, 'error: %s' % self.report_missing_param(key)
                
                self.attributes[attributes_ids_dict[key]] = args[key]
    
            # self.fix_attributes()
            
    def evaluate(self, input):
        """
            Evaluates the layer on the given input
            
            input : np.array([*])
            
            returns the normalized output
        """
        
        if not self.isempty(self.attributes[BN_TRANED_MEAN_ID]) and not self.isempty(self.attributes[BN_TRANED_VAR_ID]) and \
           not self.isempty(self.attributes[BN_OFFSET_ID]) and not self.isempty(self.attributes[BN_SCALE_ID]):
            y = input - self.isempty(self.attributes[BN_TRANED_MEAN_ID])
                                     
            for i in range(self.attributes[BN_NUM_CHANNELS_ID]):
                y[:,:,i] = np.divide(y[:,:,i], np.sqrt(self.attributes[BN_TRANED_MEAN_ID] + self.attributes[BN_EPSILON_ID]))
                y[:,:,i] = np.multiply(y[:,:,i], self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
        elif not self.isempty(self.attributes[BN_SCALE_ID]) and not self.isempty(self.attributes[BN_OFFSET_ID]) \
             and self.isempty(self.attributes[BN_TRANED_MEAN_ID]) and self.isempty(self.attributes[BN_TRANED_VAR_ID]):
            y = input - self.isempty(self.attributes[BN_TRANED_MEAN_ID])
            
            if self.attributes[BN_NUM_CHANNELS_ID] == 1:
                y = np.multiply(y, self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
            else:
                for i in range(self.attributes[BN_NUM_CHANNELS_ID]):
                    y[:,:,i] = np.multiply(y[:,:,i], self.attributes[BN_SCALE_ID]) + self.attributes[BN_OFFSET_ID]
        else:
            y = input
            
        return y
    
    def reach_single_input(self, input_image):
        """
            Performs reachability analysis on a single input image
            
            input_image : ImageStar (ImageZono) -> the input image
            
            returns the normalized image
        """
    
        assert isinstance(input_image, 'Star') or isinstance(input_image, 'ImageStar')     \
               or isinstance(input_image, 'Zono') or isinstance(input_image, 'ImageZono'), \
               'error: %s' % BN_ERRORMSG_INVALID_INPUT
               
        output = input_image
               
        if isinstance(input_image, 'ImageStar') or isinstance(input_image, 'ImageZono'):
            self.fix_input(input_image)
            
            buffer = np.zeros((1, 1, self.attributes[BN_NUM_CHANNEL_ID]))
            
            if not self.isempty(self.attributes[BN_TRANED_MEAN_ID]) and not self.isempty(self.attributes[BN_TRANED_VAR_ID]) and \
               not self.isempty(self.attributes[BN_OFFSET_ID]) and not self.isempty(self.attributes[BN_SCALE_ID]):
                for i in range(self.attributes[BN_NUM_CHANNEL_ID]):
                    buffer = 1 / np.sqrt(self.attributes[BN_TRANED_VAR_ID][0, 0, i] + self.attributes[BN_EPSILON_ID])
                    
                    output = output.affine_map(buffer, np.multiply(-buffer, self.attributes[BN_TRANED_MEAN_ID]))
                
            output = output.affine_map(self.attributes[BN_SCALE_ID], self.attributes[BN_OFFSET_ID])
            
        else:
            buffer = np.divide(self.attributes[BN_SCALE_ID], np.sqrt(self.attributes[BN_VAR_ID] + self.attributes[BN_EPSILON_ID]))
            
            new_V = output.get_V()
            for i in range(new_V.shape[1]):
                new_V[:, i] = (np.multiply(output.get_V()[:,i] - self.attributes[BN_TRANED_MEAN_ID], buffer) + self.attributes[BN_OFFSET_ID])
    
            output.set_V(new_V)
            
        return output
    
    def reach_multiple_inputs(self, input_images, option):
        """
            Performs reachability analysis on several input images
            
            input_images : [Image*] -> a set of ImageStar-s or ImageZono-s (Star-s or Zono-s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns a set of normalized images
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
        
        assert args[BN_ARGS_METHODID] < 5, 'error: %s' % BN_ERRMSG_UNK_REACH_METHOD
        
        IS = self.reach_multiple_inputs(args[BN_REACH_ARGS_NPUT_IMAGES_ID], option)    
    
########################## UTILS ##########################

    def fix_attributes(self):
        raise NotImplemented("will be implemented during testing")

    def init_empty_layer(self):
        for key in self.key_id_map.keys:
            if(key == 'Name'):
                self.attributes[attributes_ids_dict[key]] = ''
            else:
                self.attributes[attributes_ids_dict[key]] = 0
                
    def validate_keys(self, args):
        for key in args.keys:
            assert key in self.key_id_map.keys, 'error: %s' % BN_ERRMSG_UNK_KEY
            
            if key == 'Name':
                assert not isinstance(args[key], str), 'error: %s' % BN_ERRMSG_NAME_NOT_STRING

    def report_missing_param(self, param_name):
        return f'error: Parameter {param_name} missing'
    


    
        