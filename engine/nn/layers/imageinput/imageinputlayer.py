class ImageInputLayer:
    """
        The Image input layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def ImageInputLayer(self, *args):
        """
            Constructor
        """
        
        if len(args) == IIL_EMPTY_ARGS_LEN:
            self.attributes[IIL_NAME_ID] = IIL_DEFAULT_NAME
            self.attributes[IIL_INPUT_SIZE_ID] = IIL_DEFAULT_INPUT_SIZE
            self.attributes[IIL_MEAN_ID] = IIF_DEFAULT_MEAN # zero
        elif len(args) != IIL_ARGS_LEN:
            raise Exception(IIL_ERRMSG_INVALID_ARG_LEN)
        
        self.attributes = []
        
        for i in range(RELUL_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        # TODO: add assertions
        self.attributes[IIL_NAME_ID] = args[IIL_ARGS_NAME_ID]
        self.attributes[IIL_INPUT_SIZE_ID] = args[IIL_ARGS_INPUT_SIZE_ID]
        self.attributes[IIL_MEAN_ID] = args[IIL_ARGS_MEAN_ID].astype('float64')
        
    def evaluate(self, input):
        """
            Evaluates the layer using the given input
            
            input : np.array([*]) -> the input image
            returns a standardized image
        """
        
        return input.astype('float64') - self.attributes[IIL_MEAN_ID]
    
    def reach_single_input(self, input):
        """
            input : ImageStar -> the input image
            
            returns a standardized ImageStar
        """
        
        return input_image.affine_map(np.array([]), -self.attributes[IIL_MEAN_ID])
    
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
            IS = self.reach_star_multiple_inputs(args[SIGNL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGNL_REACH_ARGS_METHOD_ID], args[SIGNL_REACH_ARGS_OPTION_ID], args[SIGNL_REACH_ARGS_MODE_ID])
        else:
            raise Exception(IIL_ERRMSG_UNK_REACH_METHOD)
            
        return IS