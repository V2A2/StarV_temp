from _operator import contains
class PixelClassificationLayer:
    """
        Pixel Classification Layer object to verify segmentation networks
    """
    
    def PixelClassificationLayer(self, *args):
        """
            Constructor
            
            name : string -> the name of the layer
            classes : np.array([*]) -> a set of classes
            output_size : np.array([]) -> the output size
        """
        
        self.attributes = []
        
        for i in range(RELUL_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))

        
        if len(args) > PCL_EMPTY_ARGS_LEN:
            if len(args) == PCL_FULL_ARGS_LEN:
                self.attributes[PCL_NUM_INPUTS_ID] = args[PCL_ARGS_NUM_INPUTS_ID]
                self.attributes[PCL_INPUT_NAMES_ID] = args[PCL_ARGS_INPUT_NAMES_ID]
                self.attributes[PCL_OUTPUT_SIZE_ID] = args[PCL_ARGS_OUTPUT_SIZE_ID]
            else:
                self.attributes[PCL_NUM_INPUTS_ID] = PCL_DEFAULT_NUM_INPUTS
                self.attributes[PCL_INPUT_NAMES_ID] = PCL_DEFAULT_INPUT_NAMES
                self.attributes[PCL_OUTPUT_SIZE_ID] = PCL_DEFAULT_OUTPUT_SIZE

            assert isinstance(args[PCL_ARGS_NAME_ID], str), 'error: %s' % PCL_ERRMSG_NAME_NOT_STRING
            self.attributes[PCL_NAME_ID] = args[PCL_ARGS_NAME_ID]
            
            assert isinstance(args[PCL_ARGS_CLASSES_ID], np.ndarray), 'error: %s' % PCL_ERRMSG_CLASSES_NOT_MATRIX            
            assert isinstance(args[PCL_ARGS_OUTPUT_SIZE_ID], np.ndarray), 'error: %s' % PCL_ERRMSG_OUTPUT_SIZE_NOT_MATRIX
            assert len(args[PCL_ARGS_OUTPUT_SIZE_ID]) == 3, 'error: %s' % PCL_ERRMSG_OUTPUT_INVALID_OUTPUT_SIZE
            self.attributes[PCL_OUTPUT_SIZE_ID] = args[PCL_ARGS_OUTPUT_SIZE_ID]
        else:
            raise Exception(PCL_ERRMSG_INVALID_ARGS_NUM)
        
        a = np.unique(args[PCL_ARGS_CLASSES_ID])
        self.attributes[PCL_CLASSES_ID] = np.append(args[PCL_ARGS_CLASSES_ID], ['unknown', 'misclass'])
        
    def evaluate(self, image):
        """
            Classifies a lable for all pixels of the image
            
            image : np.array([*]) -> an output image before a softmax layer with the size of
                                     m1 x m2 x n, where n is the number of labels needed to be
                                     classified
                    
            returns a segmentation image with cass categories and a class index
        """
        
        assert len(image.shape) == 2 or len(image.shape) == 3, 'error: %s' % PCL_ERRMSG_EVAL_INVALID_INPUT_SIZE
        
        [max_id, max_val] = max(enumerate(image), key=operator.itemgetter(1))
        
        S = np.empty((image.shape[0], image.shape[1]))
        
        classes = self.attributes[PCL_CLASSES_ID]
        
        X = np.zeros((image.shape[0], image.shape[1]))
         
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                S[i, j] = classes[max_val[i,j]]
                X[i,j] = max_val[i,j]
                
        seg_im_cat = np.unique(S)
        seg_im_id = X
        
        return seg_im_id, seg_im_cat
        
    def reach_star_single_input(self, input):
        """
            Performs reachability analysis on the given ImageStar
            
            input : ImageStar -> the input set
            
            returns a segmentation image with a class index
        """
        
        h = input.get_height()
        w = input.get_width()
        seg_im_id = np.zeros((h,w))
        
        im_lb, im_ub = input.estimate_ranges()
        
        for i in range(h):
            for j in range(w):
                max_xmin = max(im_lb[i,j,:])
                pc = np.argwhere(im_ub[i,j,:] >= max_xmin)
                
                if len(pc) == 1:
                    seg_im_id[i,j] = pc
                else:
                    seg_im_id[i,j] = len(self.attributes[PCL_CLASSES_ID]) - 1
                    
        return seg_im_id
    
    def reach_relax_star_single_input(self, inpu, method, relaxed_factor):
        """
            Performs relaxed reachability analysis on the given ImageStar
            
            input : ImageStar -> the given input
            method : string -> relax-star reachability method
            relaxed_factor : float
            
            returns a segmentation image with a class index
        """
        
        h = input.get_height()
        w = input.get_width()
        nc = input.get_num_channel()
        seg_im_id = np.zeros((h,w))
        
        s = input.to_star()
        lb, ub = s.estimate_ranges()
        
        n1 = np.round((1 - relaxed_factor) * len(lb))
        
        if method == 'relax-star-range':
            [min_id, _] = -np.sort(-(ub - lb)) # TODO: check what matlab outputs here
            map = min_id[0:n1]
            
            mins = map
            maxs = map
        elif method == 'relax-star-random':
            midx = np.random.permutation(len(ub), n1) # TODO: figure out how to limit the number of values
            midx = np.transpose(midx)
            
            mins = midx
            maxs = midx
            
        elif method == 'relax-star-area':
            areas = 0.5 * np.multiply(np.abs(ub), np.abs(lb))
            [min_id, _] = -np.sort(-(areas))
            map = midx[0:n1]
            
            mins = map
            maxs = map
        elif method == 'relax-star-bound':
            N = len(ub)
            lu = np.vstack((ub, np.abs(lb)))
            [min_id, _] = -np.sort(-lu)
            midx1 = midx[0:2 * n1]
            mins = midx1[midx1 <= N]
            maxs = midx1[midx1 > N] - N
        else:
            raise Exception(PCL_ERRMSG_UNKOWN_REACH_METHOD)
            
        xmin = s.get_mins(mins, 'single')
        xmax = s.get_maxs(maxs, 'single')
        lb[map] = xmin
        ub[map] = xmax
        
        im_lb = np.reshape(lb, (h,w,nc))
        im_ub = np.reshape(ub, (h,w,nc))
        
        for i in range(h):
            for j in range(w):
                max_xmin = max(im_lb[i,j,:])
                pc = np.argwhere(im_ub[i,j,:] >= max_xmin)
                
                if len(pc) == 1:
                    seg_im_id[i,j] = pc
                else:
                    seg_im_id[i,j] = len(self.attributes[PCL_CLASSES_ID]) - 1
                    
        return seg_im_id

    def reach_star_multiple_inputs(self, input, option):
        """
            Performs reachability analysis for a set of ImageStar-s
            
            input : ImageStar* -> a set of ImageStar-s
            option:
            
            returns segmentation images with class indices
        """
        
        seg_ims_ids = np.empty((len(input), 1))
        
        for i in range(len(input)):
            seg_ims_ids[i] = self.reach_star_single_input(input[i])
            
        return seg_ims_ids
    
    def reach_relax_star_multiple_inputs(self, input, method, relaxed_factor, option):
        """
            Performs relaxed reachability analysis for a set of ImageStar-s
            
            input : ImageStar* -> a set of ImageStar-s
            option:
            
            returns segmentation images with class indices
        """
        
        seg_ims_ids = np.empty((len(input), 1))
        
        for i in range(len(input)):
            seg_ims_ids[i] = self.reach_relax_star_single_input(input[i], method, relaxed_factor)
            
        return seg_ims_ids
            
            
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
        
        if method == 'approx-star' or method == 'exact-star' or method == 'abs-dom':
            IS = self.reach_star_multiple_inputs(args[SIGNL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGNL_REACH_ARGS_METHOD_ID], args[SIGNL_REACH_ARGS_OPTION_ID], args[SIGNL_REACH_ARGS_MODE_ID])
        elif contains(method, 'relax-star'):
            IS = self.reach_relax_star_multiple_inputs(args[SIGNL_REACH_ARGS_INPUT_IMAGES_ID], args[SIGNL_REACH_ARGS_METHOD_ID], args[SIGNL_REACH_ARGS_OPTION_ID], args[SIGNL_REACH_ARGS_MODE_ID])
        else:
            raise Exception(PCL_ERRMSG_UNK_REACH_METHOD)
            
        return IS