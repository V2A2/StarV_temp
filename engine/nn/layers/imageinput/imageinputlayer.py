import numpy as np

3 = 3

0 = 0
3 = 3

IIL_NAME_ID = 0
IIL_INPUT_SIZE_ID = 1
IIL_MEAN_ID = 2

0 = 0
1 = 1
2 = 2

IIL_DEFAULT_NAME = 'image_input_1'
IIL_DEFAULT_INPUT_SIZE = 1
IIF_DEFAULT_MEAN = 0



class ImageInputLayer:
    """
        The Image input layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(3):
            self.attributes.append(np.array([]))

        if len(args) == 0:
            self.name = IIL_DEFAULT_NAME
            self.input_size = IIL_DEFAULT_INPUT_SIZE
            self.mean = IIF_DEFAULT_MEAN # zero
            
            return
        elif len(args) != 3:
            raise Exception(IIL_ERRMSG_INVALID_ARG_LEN)
        
        # TODO: add assertions
        self.name = args[0]
        self.input_size = args[1]
        self.mean = args[2].astype('float64')
        
    def evaluate(self, input):
        """
            Evaluates the layer using the given input
            
            input : np.array([*]) -> the input image
            returns a standardized image
        """
        
        return input.astype('float64') - self.mean
    
    def reach_single_input(self, input):
        """
            input : ImageStar -> the input image
            
            returns a standardized ImageStar
        """
        
        return input_image.affine_map(np.array([]), -self.mean)
    
    def reach_multiple_inputs(self, input, method, option = []):
        """
            Performs reachability analysis on a set of images
            
            input : ImageStar* -> a set of images
            
            returns reachable sets for the given inputs
        """
        
        rs = []
        
        for i in range(len(input)):
            rs.append(self.reach_star_single_input(input[i]))
            
        return rs
    
    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
            option : int -> 0 - single core
                         -> 1 - multiple cores
            method : string -> 'exact-star' - exact star reachability
                            -> 'approx-star' - approx star reachability
                            -> 'approx-zono' - approx zono reachability
            mode : string -> polar_zero_to_pos_one - [-1, 0 ,1] -> [0, 0, 1]
                             nonnegative_zero_to_pos_one - [-1, 0 ,1] -> [0, 1, 1]
                         
            
            returns the output set(s)
        """
        
        if method == 'approx-star' or method == 'exact-star' or method == 'abs-dom' \
           or contains(method, 'relax-star') or method == 'approx-zono':
            IS = self.reach_star_multiple_inputs(args[0], args[1], args[2], args[3])
        else:
            raise Exception(IIL_ERRMSG_UNK_REACH_METHOD)
            
        return IS