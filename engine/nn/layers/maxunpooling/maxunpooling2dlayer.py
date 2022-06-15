MAXUNP2D_ERRMSG_INVALID_INPUTS_NUM_ID = 'Invalid number of inputs (should be at least one)'
MAXUNP2D_ERRMSG_INVALID_OUTPUTS_NUM_ID = 'Invalid number of outputs (should be at least one)'
MAXUNP2D_ERRMSG_INVALID_NUMER_OF_INPUTS = 'Invalid number of inputs (should be 0, 2, 3, or 5)'
MAXUNP2D_ERRMSG_UNK_REACH_METHOD = 'Unknown reachability method'

MAXUNP2D_EMPTY_ARGS_LEN = 0
MAXUNP2D_INPUTS_ARGS_LEN = 2
MAXUNP2D_INPUTS_OUTPUTS_ARGS_LEN = 5

MAXUNP2D_INPUTS_ARGS_OFFSET = 1

MAXUNP2D_NAME_ID = 0
MAXUNP2D_INPUTS_NUM_ID = 1
MAXUNP2D_INPUTS_NAMES_ID = 2
MAXUNP2D_OUTPUTS_NUM_ID = 3
MAXUNP2D_OUTPUTS_NAMES_ID = 4
MAXUNP2D_PAIRED_MP_NAME_ID = 5

MAXUNP2D_ARGS_NAME_ID = 0
MAXUNP2D_ARGS_INPUTS_NUM_ID = 1
MAXUNP2D_ARGS_INPUTS_NAMES_ID = 2
MAXUNP2D_ARGS_OUTPUTS_NUM_ID = 3
MAXUNP2D_ARGS_OUTPUTS_NAMES_ID = 4

MAXUNP2D_INPUTS_ARGS_OFFSET = 'max_unpooling_2d_layer'
MAXUNP2D_DEFAULT_DISPLAY_OPTION = []



class MaxUnpooling2DLayer:
    """
        The MaxUnPooling 2D layer class in CNN
        Contains constructor and reachability analysis methods
    """
    
    def MaxUnpooling2DLayer(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(MAXUNP2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))

        
        if len(args) > MAXUNP2D_EMPTY_ARGS_LEN:
            if len(args) == MAXUNP2D_INPUTS_ARGS_LEN:
                args = self.offset_args(args, MAXUNP2D_INPUTS_ARGS_OFFSET)
                self.attributes[MAXUNP2D_NAME_ID] = MAXUNP2D_DEFAULT_NAME
                
            assert args[MAXUNP2D_ARGS_INPUTS_NUM_ID] > 0, 'error: %s' % MAXUNP2D_ERRMSG_INVALID_INPUTS_NUM_ID
            
            self.attributes[MAXUNP2D_INPUTS_NUM_ID] = args[MAXUNP2D_ARGS_INPUTS_NUM_ID]
            self.attributes[MAXUNP2D_INPUTS_NAMES_ID] = args[MAXUNP2D_ARGS_INPUTS_NAMES_ID]
                
            if len(args) == MAXUNP2D_INPUTS_OUTPUTS_ARGS_LEN:
                assert args[MAXUNP2D_ARGS_OUTPUTS_NUM_ID] > 0, 'error: %s' % MAXUNP2D_ERRMSG_INVALID_OUTPUTS_NUM_ID
                
                self.attributes[MAXUNP2D_OUTPUTS_NUM_ID] = args[MAXUNP2D_ARGS_OUTPUTS_NUM_ID]
                self.attributes[MAXUNP2D_OUTPUTS_NAMES_ID] = args[MAXUNP2D_ARGS_OUTPUTS_NAMES_ID]
        elif len(args) == MAXUNP2D_EMPTY_ARGS_LEN:
            self.attributes[MAXUNP2D_NAME_ID] = MAXUNP2D_DEFAULT_NAME
            self.attributes[MAXUNP2D_INPUTS_NUM_ID] = MAXUNP2D_DEFAULT_INPUTS_NUM
            self.attributes[MAXUNP2D_INPUTS_NAMES_ID] = MAXUNP2D_DEFAULT_INPUTS_NAMES_NUM
            
            self.attributes[MAXUNP2D_OUTPUTS_NUM_ID] = MAXUNP2D_DEFAULT_OUTPUTS_NUM
            self.attributes[MAXUNP2D_OUTPUTS_NAMES_ID] = MAXUNP2D_DEFAULT_OUTPUTS_NAMES_NUM
        else:
            raise Exception(MAXUNP2D_ERRMSG_INVALID_NUMER_OF_INPUTS)
    
    @staticmethod
    def evaluate(input, max_indices, output_size):
        """
            Evaluates the layer on the given input
            
            input : np.array([*]) -> the input array
            indices : np.array([]) -> max-point indices
            output_size : np.array([]) -> the output size of the unpooled image
            
            returns an unpooled image
        """

        unpool = nn.MaxUnpool2D(2, stride = 2)
        
        return unpool(input, max_indices, output_size=output_size).cpu().detach().numpy()
    
    @staticmethod
    def stepReach_star_single_input(input, max_points, V, lb, ub):
        """
            Performs a stepReach operation on the given ImageStar
            
            input : ImageStar -> the input ImageStar
            max_points : np.array([]) -> max-point indices
            V : np.array([*]) -> max-point star set's V
            lb : np.array([]) -> max-point star set's lb
            ub : np.array([]) -> max-point star set's ub
            
            returns a stepReach operation result
        """
        result = []
        max_points_num = max_points.shape[0]
        
        for i in range(max_points):
            max_point = max_points[i, :, :]
            
            R1 = ImageStar(input.get_V(), input.get_C(), input.get_d(), input.get_pred_lb(), input.get_pred_ub(), input.get_im_lb, input_get_im_ub())
            
            R1.set_max_ids(input.get_max_ids())
            R1.set_input_sizes(input.get_input_sizes())
            
            current_V = R1.get_V()
            current_V[i,j,k,:] = V
            R1.set_V(current_V)
            
            current_im_lb = R1.get_im_lb()
            current_im_ub = R1.get_im_ub()
            current_im_lb[i,j,k] = lb
            current_im_ub[i,j,k] = ub
            R1.set_im_lb(current_im_lb)
            R1.set_im_ub(current_im_ub)
            
            R.append(R1)
        
        return R
        
    def stepReach_multiple_inputs(self, input, max_points, V, lb, ub):
        """
            Performs the stepReach operation on the set of images
            
            input : ImageStar* -> a set of input images
            max_points : np.array([np.array([*])*]) -> max-point indices
            V : np.array([*]) -> max-point star set's V
            lb : np.array([]) -> max-point star set's lower bound
            ub : np.array([]) -> max-point star set's upper bound
            
            returns the results of applying the stepReach operation to the given images
        """
        
        result = []
        for i in range(len(input)):
            result.append(self.stepReach_star_single_input(input[i], max_points, V, lb, ub))
            
        return result
    
    def reach_star(self, input):
        """
            input : ImageStar -> the input image
            
            returns the reachable input
        """
        
        new_max_indices = input.get_max_ids()
        new_input_sizes = input.get_input_sizes()
        input_size = []
        max_id = []
        
        for i in range(len(new_max_indices)):
            if new_max_indices[i].get_name() == self.attributes[MAXUNP2D_PAIRED_MP_NAME_ID]:
                max_id = new_max_ids[i].get_max_id()
                input_size = new_input_sizes[i].get_input_size()
                
                new_max_ids[i] = []
                new_input_sizes[i] = []
                break
            
        num_channel = input.get_num_channel()
        num_pred = input.get_num_pred()
        
        h = input.get_height()
        w = input.get_width()
        
        if self.isempty(input.get_im_lb()):
            input.estimate_ranges()
            
        new_V = np.zeros((input_size[0], input_size[1], num_channel, num_pred + 1))
        new_im_lb = np.zeros((input_size[0], input_size[1], num_channel))
        new_im_ub = np.zeros((input_size[0], input_size[1], num_channel))
        
        R0 = ImageStar(new_V, input.get_C(), input.get_d(), input.get_pred_lb(), input.get_pred_ub(), new_im_lb, new_im_ub)
        R0.set_max_ids(new_max_indices)
        R0.set_input_sizes(new_input_sizes)
        
        R = R0
        
        for k in range(num_channel):
            for i in range(h):
                for j in range(w):
                    current_max_id = max_id[i,j,k]
                    
                    max_points = np.hstack(current_max_id, k * np.ones((current_max_id.shape[0], 1)))
                    V1 = input.get_V()[i,j,k,:]
                    lb = input.get_im_lb[i,j,k]
                    ub = input.get_im_ub[i,j,k]
                    R = self.stepReach_multiple_inputs(R, max_points, V1, lb, ub)
                    
        return R
        
    def reach_star_multiple_inputs(self, input_images, option = MAXUNP2D_DEFAULT_DISPLAY_OPTION):
        """
            Performs reachability analysis on multiple images
            
            input_images : ImageStar* -> the input images
            option
            
            return the set of reachable inputs
        """
        
        result = []
        
        for i in range(len(input_images)):
            result.append(self.reach_star(input_images[i]))
            
        return result
    
    def reach(self, *args):
        """
            Performs reachability analysis for the given images
            
            input_images : Image* -> the input ImageStar or ImageZono
            method : string -> reachability method
            
            returns reachable sets for the given inputs
        """
        
        if method == 'approx-star' or method == 'exact-star' or method == 'approx-zono':
            return self.reach_star_multiple_inputs(input_images)
        else:
            raise Exception(MAXUNP2D_ERRMSG_UNK_REACH_METHOD)
        