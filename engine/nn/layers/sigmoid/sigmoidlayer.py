SIGMOIDL_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'
SIGMOIDL_ERRORMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 1, 5)'
SIGMOIDL_ERRORMSG_INVALID_INPUT = 'The input should be either an ImageStar or ImageZono'

SIGMOIDL_ATTRIBUTES_NUM = 5

SIGMOIDL_FULL_ARGS_LEN = 5
SIGMOIDL_NAME_ARGS_LEN = 1
SIGMOIDL_EMPTY_ARGS_LEN = 0

SIGMOIDL_NAME_ID = 0
SIGMOIDL_NUM_INPUTS_ID = 1
SIGMOIDL_INPUT_NAMES_ID = 2
SIGMOIDL_NUM_OUTPUTS_ID = 3
SIGMOIDL_OUTPUT_NAMES_ID = 4

SIGMOIDL_NAME_ARGS_ID = 0
SIGMOIDL_NUM_INPUTS_ARGS_ID = 1
SIGMOIDL_INPUT_NAMES_ARGS_ID = 2
SIGMOIDL_NUM_OUTPUTS_ARGS_ID = 3
SIGMOIDL_OUTPUT_NAMES_ARGS_ID = 4

SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID = 0
SIGMOIDL_REACH_ARGS_OPTION_ID = 1
SIGMOIDL_REACH_ARGS_METHOD_ID = 2
SIGMOIDL_REACH_ARGS_RELAX_FACTOR_ID = 3

SIGMOIDL_DEFAULT_NAME = 'relu_layer'

class SigmoidLayer:
    """
        The Sigmoid layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def SigmoidLayer(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(SIGMOIDL_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
        
        if len(args) == SIGMOIDL_FULL_ARGS_LEN:
            self.attributes[SIGMOIDL_NAME_ID] = self.attributes[SIGMOIDL_ARGS_NAME_ID]
            self.attributes[SIGMOIDL_NUM_INPUTS_ID] = self.attributes[SIGMOIDL_ARGS_NUM_INPUTS_ID]
            self.attributes[SIGMOIDL_INPUT_NAMES_ID] = self.attributes[SIGMOIDL_ARGS_INPUT_NAMES_ID]
            self.attributes[SIGMOIDL_NUM_OUTPUTS_ID] = self.attributes[SIGMOIDL_ARGS_NUM_OUTPUTS_ID]
            self.attributes[SIGMOIDL_OUTPUT_NAMES_ID] = self.attributes[SIGMOIDL_ARGS_OUTPUT_NAMES_ID]
        elif len(args) == SIGMOIDL_NAME_ARGS_LEN:
            assert isinstance(self.attributes[SIGMOIDL_ARGS_NAME_ID], str), 'error: %s' % SIGMOIDL_ERRMSG_NAME_NOT_STRING

            self.attributes[SIGMOIDL_NAME_ID] = self.attributes[SIGMOIDL_ARGS_NAME_ID]
        elif len(args) == SIGMOIDL_EMPTY_ARGS_LEN:
            self.attributes[SIGMOIDL_NAME_ID] = SIGMOIDL_DEFAULT_NAME
        else:
            raise Exception(SIGMOIDL_ERRORMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(_, input):
        """
            Evaluates the layer on the given input
            input : np.array([*]) -> a 2- or 3-dimensional array
            
            returns the result of apllying ReLU activation to the given input
        """
            
        return np.reshape(LogSig.evaluate(torch.reshape(input,(np.prod(input.shape), 1))), input.shape)
    
    def reach_star_single_input(_, input, method, relax_factor):
        """
            Performs reachability analysis on the given input
                 
            input : ImageStar -> the input ImageStar
            method : string -> reachability method
            relax_factor : double -> relaxation factor for over-approximate star reachability
                
            returns a set of reachable sets for the given input images
        """
             
        assert isinstance(input, ImageStar), 'error: %s' % SIGMOIDL_ERRORMSG_INVALID_INPUT
            
        reachable_sets = LogSig.reach(input_image.to_star(), method, [], relax_factor)

        rs = []
        
        for i in range(len(reachable_sets)):
            rs.append(reachable_sets[i].to_image_star(h, w, c))
            
        return rs
    
    def reach_star_multiple_inputs(self, input_images, method, option, relax_factor):
        """
            Performs reachability analysis on the given multiple inputs
            
            input_images : ImageStar* -> a set of input images
            method : string -> 'exact-star' - exact reachability analysis
                               'approx-star' - overapproximate reachability analysis
            option
            relax_factor : float -> a relaxation factor for overapproximate star reachability
            
            returns an array of reachable sets for the given input images
        """
    
        r_images = []
        
        for i in range(len(input_images)):
            r_images.append(self.reach_star_single_input(input_images[i], method, relax_factor))
            
        return r_images
        
    def reach_zono(self, input_image):
        """
            Performs reachability analysis on the given input using zonotope
            
            input_image : ImageZono -> the input set
            
            returns a reachable set or the given ImageZono
        """    
        
        assert isinstance(input, ImageZono), 'error: %s' % SIGMOIDL_ERRORMSG_INVALID_INPUT
        
        reachable_set = LogSig.reach(input_image.toZono())
        return reachable_set.toImageZono(input_image.get_height(), \
                                         input_image.get_width(), \
                                         input_image.get_channel())
        
    def reach_zono_multiple_inputs(self, input_images, option):
        """
            Performs reachability analysis on the given set of ImageZono-s
            
            input_images : ImageZono* -> a set of images
            
            returns reachable sets of the input images
        """
        
        rs = []
        for i in range(len(input_images)):
            rs.append(self.reach_zono(input_images[i]))

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
                         
            
            returns the output set(s)
        """
        
        if method == 'approx-star' or method == 'exact-star':
            IS = self.reach_star_multiple_inputs(args[SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGMOIDL_REACH_ARGS_METHOD_ID], args[SIGMOIDL_REACH_ARGS_OPTION_ID], args[SIGMOIDL_REACH_ARGS_RELAX_FACTOR_ID])
        elif method == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[SIGMOIDL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGMOIDL_REACH_ARGS_OPTION_ID])
        else:
            raise Exception(SIGMOIDL_ERRMSG_UNK_REACH_METHOD)
            
        return IS
