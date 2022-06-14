import numpy as np
import torch
import torch.nn as nn 

MAXP2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
MAXP2D_ERRMSG_INVALID_POOL_SIZE = 'Invalid pool size'
MAXP2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
MAXP2D_ERRMSG_INVALID_PADDING_SIZE = 'Invalide padding size'
MAXP2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 3, or 4)'
MAXP2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
MAXP2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
MAXP2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'

MAXP2D_ERRMSG_RE_SINGLE_INVALID_ARGS_NUM = 'Invalid numer of arguments to perform reachability analysis'

MAXP2D_ERRORMSG_INVALID_IMGS_INPUT = 'The input image should be an ImageStar'
MAXP2D_ERRORMSG_INVALID_IMGZ_INPUT = 'The input image should be an ImageZono'

MAXP2D_ERRMSG_INVALID_SPLIT_INDEX = 'Invalid split index, it should have 3 columns and at least 1 row'


MAXP2D_ATTRIBUTES_NUM = 8

MAXP2D_FULL_ARGS_LEN = 8
MAXP2D_FULL_CALC_ARGS_LEN = 4
MAXP2D_CALC_ARGS_LEN = 3

MAXP2D_NAME_ID = 0
MAXP2D_POOL_SIZE_ID = 1
MAXP2D_STRIDE_ID = 2
MAXP2D_PADDING_SIZE_ID = 3
MAXP2D_NUMINPUTS_ID = 4
MAXP2D_NUMOUTPUTS_ID = 5
MAXP2D_INPUTNAMES_ID = 6
MAXP2D_OUTPUTNAMES_ID = 7

MAXP2D_ARGS_NAME_ID = 0
MAXP2D_ARGS_POOL_SIZE_ID = 1
MAXP2D_ARGS_STRIDE_ID = 2
MAXP2D_ARGS_PADDING_SIZE_ID = 3
MAXP2D_ARGS_NUMINPUTS_ID = 4
MAXP2D_ARGS_NUMOUTPUTS_ID = 5
MAXP2D_ARGS_INPUTNAMES_ID = 6
MAXP2D_ARGS_OUTPUTNAMES_ID = 7

MAXP2D_EVAL_ARGS_INPUT_ID = 0
MAXP2D_RE_MULT_ARGS_INPUT_ID = 0
MAXP2D_RE_SINGLE_ARGS_INPUT_ID = 0
MAXP2D_STEP_SPLIT_ARGS_INPUT_IMGS_ID = 0
MAXP2D_STEP_SPLIT_ARGS_ORI_IMG_ID = 1
MAXP2D_STEP_SPLIT_ARGS_POS_ID = 2
MAXP2D_STEP_SPLIT_ARGS_SPLIT_INDEX_ID = 3
MAXP2D_RA_MULT_INPUT_ID = 0
MAXP2D_RZ_MULT_INPUT_ID = 0

MAXP2D_EVAL_FULL_ARGS_LEN = 2
MAXP2D_EVAL_ARGS_LEN = 1

MAXP2D_REACH_ARGS_INPUT_IMAGES_ID = 0

MAXP2D_DEFAULT_LAYER_NAME = 'average_pooling_layer'
MAXP2D_DEFAULT_POOL_SIZE = np.array([2,2])
MAXP2D_DEFAULT_STRIDE = np.array([1,1])
MAXP2D_DEFAULT_PADDING_SIZE = np.array([0,0,0,0])

DEFAULT_DISPLAY_OPTION = []

class MaxPooling2DLayer:
    """
        The MaxPooling 2D layer class in CNN
        Contain constructor and reachability analysis methods
        Main references:
        1) An intuitive explanation of convolutional neural networks: 
           https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/
        2) More detail about mathematical background of CNN
           http://cs231n.github.io/convolutional-networks/
           http://cs231n.github.io/convolutional-networks/#pool
        3) Matlab implementation of Convolution2DLayer and MaxPooling (for training and evaluating purpose)
           https://www.mathworks.com/help/deeplearning/ug/layers-of-a-convolutional-neural-network.html
           https://www.mathworks.com/help/deeplearning/ref/nnet.cnn.layer.maxpooling2dlayer.html
    """
    
    def MaxPooling2DLayer(self, *args):
        self.attributes = []       
        
        for i in range(MAXP2D_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) <= MAXP2D_FULL_ARGS_LEN and len(args) > 0:
            if len(args == MAXP2D_FULL_ARGS_LEN):
                self.attributes[MAXP2D_NUMINPUTS_ID] = args[MAXP2D_ARGS_NUMINPUTS_ID]
                self.attributes[MAXP2D_NUMOUTPUTS_ID] = args[MAXP2D_ARGS_NUMOUTPUTS_ID]
                self.attributes[MAXP2D_INPUTNAMES_ID] = args[MAXP2D_ARGS_INPUTNAMES_ID]
                self.attributes[MAXP2D_OUTPUTNAMES_ID] = args[MAXP2D_ARGS_OUTPUTNAMES_ID]
            elif len(args) == MAXP2D_FULL_CALC_ARGS_LEN:
                assert isinstance(args[MAXP2D_ARGS_NAME_ID], str), 'error: %s' % MAXP2D_ERRMSG_NAME_NOT_STRING
                self.attributes[MAXP2D_NAME_ID] = args[MAXP2D_ARGS_NAME_ID]

            if len(args) == MAXP2D_CALC_ARGS_LEN:
                args = self.offset_args(args, MAXP2D_CALC_ARGS_OFFSET)
                self.attributes[MAXP2D_NAME_ID] = MAXP2D_DEFAULT_LAYER_NAME                
                
            if self.isempty(args[MAXP2D_ARGS_POOL_SIZE_ID]):
                self.attributes[MAXP2D_POOL_SIZE_ID] = MAXP2D_DEFAULT_POOL_SIZE
                self.attributes[MAXP2D_STRIDE_ID] =  MAXP2D_DEFAULT_STRIDE
                self.attributes[MAXP2D_PADDING_SIZE_ID] = MAXP2D_DEFAULT_PADDING_SIZE
            else:
                
                assert isinstance(args[MAXP2D_ARGS_POOL_SIZE_ID], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert len(args[MAXP2D_ARGS_POOL_SIZE_ID].shape[0]) == 1 or \
                        len(args[MAXP2D_ARGS_POOL_SIZE_ID].shape[1]) == 2 or \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_POOL_SIZE
                    
                assert isinstance(args[MAXP2D_ARGS_STRIDE_ID], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert len(args[MAXP2D_ARGS_STRIDE_ID].shape[0]) == 1 or \
                        len(args[MAXP2D_ARGS_STRIDE_ID].shape[1]) == 2 or \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_STRIDE
                            
                assert isinstance(args[MAXP2D_ARGS_PADDING_SIZE_ID], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert len(args[MAXP2D_ARGS_PADDING_SIZE_ID].shape[0]) == 1 or \
                        len(args[MAXP2D_ARGS_PADDING_SIZEE_ID].shape[1]) == 4 or \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_PADDING_SIZE
                            
                self.attributes[MAXP2D_POOL_SIZE_ID] = args[MAXP2D_ARGS_POOL_SIZE_ID]
                self.attributes[MAXP2D_STRIDE_ID] = args[MAXP2D_ARGS_STRIDE_ID]
                self.attributes[MAXP2D_PADDING_SIZE_ID] = args[MAXP2D_ARGS_PADDING_SIZE_ID]
        elif len(args) == 0:
                self.attributes[MAXP2D_NAME_ID] = MAXP2D_DEFAULT_LAYER_NAME                

                self.attributes[MAXP2D_POOL_SIZE_ID] = args[MAXP2D_ARGS_POOL_SIZE_ID]
                self.attributes[MAXP2D_STRIDE_ID] = args[MAXP2D_ARGS_STRIDE_ID]
                self.attributes[MAXP2D_PADDING_SIZE_ID] = args[MAXP2D_ARGS_PADDING_SIZE_ID]

        else:       
            raise Exception(MAXP2D_ERRMSG_INVALID_NUMBER_OF_INPUTS)
        
    def evaluate(self, *args):
        """
            Applies max pooling2D operation using pytorch functionality
            
            input : np.array([*]) -> a 3D array
            NOT IMPLEMENTED YET => option : str -> 'single' - single precision of computation
                                   'double' - double precision of computation
                                   'empty' - empty precision of computation
                            
            returns the pooled output
        """
        
        current_option = 'double'
        
        if len(args) == MAXP2D_EVAL_FULL_ARGS_LEN:
            assert args[MAXP2D_EVAL_PRECISION_OPT_ID] == 'single' or \
                   args[MAXP2D_EVAL_PRECISION_OPT_ID] == 'double' or \
                   args[MAXP2D_EVAL_PRECISION_OPT_ID] == 'empty', \
                   'error: %s' % MAXP2D_ERRMSG_INVALID_PRECISION_OPT
        elif len(args) != MAXP2D_EVAL_ARGS_LEN:
            raise(MAXP2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        avgpool = nn.MaxPool2d(kernel_size=self.attributes[MAXP2D_POOL_SIZE_ID].shape, \
                             stride=self.attributes[MAXP2D_STRIDE_ID].shape, \
                             padding=self.attributes[MAXP2D_PADDING_SIZE_ID].shape)
        
    
            
        return avgpool(args[MAXP2D_EVAL_ARGS_INPUT_ID]).cpu().detach().numpy()
    
    def reach_star_exact_multiple_inputs(self, *args):
        """
            Performs exact reachability analysis on multiple input images
            
            input_images : Image* -> the input images
            dis_opt : string -> display option
            
            returns the reachable sets for the given images
        """
        
        rs = []
        
        input_images = args[MAXP2D_RE_MULT_ARGS_INPUT_ID]
        
        for i in range(len(input_images)):
            rs.append(self.reach_star_exact(input_images[i]))
            
        return rs
    
    def reach_exact_single_input(self, *args):
        """
            Performs exact reachability analysis on a single input image
            
            input_image : ImageStar -> the input image
            dis_opt : string -> display option
            
            returns the reachable set for the given image
        """
    
        assert len(args) >= 2, 'error: %s' % MAXP2D_ERRMSG_RE_SINGLE_INVALID_ARGS_NUM
        
        assert isinstance(input_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
               
        input_image = args[MAXP2D_RE_SINGLE_ARGS_INPUT_ID]
               
        start_points = self.get_start_points(input_image.get_V()[:, :, 0, 0])
        h, w = self.get_size_maxMap(input_image.get_V()[:, :, 0, 0])
        
        padded_image = self.get_zero_padding_imageStar(input_image)
        
        max_index = numpy.empty((h, w, padded_image.get_num_channel())) 
        max_index_result = max_index
        
        maxMap_basis_V = np.zeros((h, w, padded_image.get_num_channel(), padded_image.get_num_pred() + 1))
        
        split_pos = []
        
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_index[i,j,k] = padded_image.get_local_max_index(start_points[i,j], \
                                                                        self.attributes[MAXP2D_ARGS_POOL_SIZE_ID], \
                                                                        k)
                    
                    if max_index[i,j,k].shape[0] == 0:
                        maxMap_basis_V[i,j,k, :] = padded_image.get_V()[max_index[i,j,k][0], max_index[i,j,k][1], k, :]
                        max_index_result[i,j,k] = max_index[i,j,k]
                    else:
                        split_pos.append([[i,j,k]])
                        
        current_split_size = split_pos.shape[0]
        
        if dis_opt == DEFAULT_DISPLAY_OPTION:
            print(MAXP2D_MSG_SPLITS_OCCURRED)
            
        images = ImageStar(maxMap_basis_V, padded_image.get_C(), padded_image.get_d(), padded_image.get_pred_lb(),padded_image.get_pred_ub())
        images.add_max_idx(self.attributes[MAXP2D_NAME_ID], max_index_result)
        images.add_input_size(self.attributes[MAXP2D_NAME_ID], np.array([padded_image.get_height(), padded_image.get_width()]))
        
        if current_split_size > 0:
            for i in range(current_split_size):
                images_num = len(images)
                images = self.step_split_multiple_inputs(images, padded_image, split_pos[i, :, :], max_index[split_pos[i, 0], split_pos[i, 1], split_pos[i, 2]], [])
                
                images_num_post_split = len(images)
                
                if dis_opt == DEFAULT_DISPLAY_OPTION:
                    print(MAXP2D_MSG_SPLIT_DETAILS)
            
    
    def get_start_points(self, input_image):
        """
            Computes a collection of start points for maxMap
            
            input : Image -> the input ImageStar or ImageZono
            
            returns a set of start points of maxMap
        """
    
        I = self.get_zero_padding_input(input_image)
        
        h, w = self.get_size_maxMap(input_image, I)
        
        start_points = np.empty((h, w))
        
        for i in range(h):
            for j in range(w):
                start_points[i, j] = np.zeros((1, 2))
                
                if i == 1:
                    start_points[i, j][0] = 1
                if j == 1:
                    start_points[i, j][1] = 1
                    
                if i > 1:
                    start_points[i, j][0] = start_points[i - 1, j][0] + self.attributes[MAXP2D_STRIDE_ID][0]
                if j > 1:
                    start_points[i, j][1] = start_points[i, j - 1][1] + self.attributes[MAXP2D_STRIDE_ID][1]
                    
        return start_points
        
    def get_size_maxMap(self, input_image, padded_image = []):
        """
            Computes the height and width of the maxMap
            Reference: http://cs231n.github.io/convolutional-networks/#pool
            
            input : Image -> the input ImageStar or ImageZono
            padded_image (optional) : Image -> padded image; reduces computational time
            
            returns the computed height and width
        """
        
        if self.isempty(padded_image):
            padded_image = self.get_zero_padding_input(input)
            
        input_size = padded_image.shape
        
        h = np.floor((input_size[0] - self.attributes[MAXP2D_STRIDE_ID][0]) / self.attributes[MAXP2D_STRIDE_ID][0] + 1)
        w = np.floor((input_size[1] - self.attributes[MAXP2D_STRIDE_ID][1]) / self.attributes[MAXP2D_STRIDE_ID][1] + 1)
    
        return h, w
    
    def get_zero_padding_input(self, input_image):
        """
            Applies zero padding to the image
            
            input : Image -> the input ImageStar or ImageZono
            
            returns the  padded image
        """
    
        input_size = input_image.shape
        
        t = self.attributes[MAXP2D_PADDING_SIZE_ID][0]
        b = self.attributes[MAXP2D_PADDING_SIZE_ID][1]
        l = self.attributes[MAXP2D_PADDING_SIZE_ID][2]
        r = self.attributes[MAXP2D_PADDING_SIZE_ID][3]
    
        if len(input_shape) == 2:
            h = input_shape[0]
            w = input_shape[1]
            
            padded_image = np.zeros((t + h + b, l + w + r))
            padded_image[t + 1 : t + h + 1, l + 1 : l + w + 1] = input_image
        elif len(input_shape) > 2:
            h = input_shape[0]
            w = input_shape[1]
            d = input_shape[2]
            
            padded_image = np.zeros((t + h + b, l + w + r, d))
    
            for i in range(d):
                padded_image[t + 1 : t + h + 1, l + 1 : l + w + 1, i] = input_image[:, :, i]
        else:
            raise Exception(MAXP2D_ERRMSG_ZERO_PAD_INVALID_INPUT)
        
    def get_zero_padding_imageStar(self, input_image):
        """
            Computes a zero-padded ImageStar or ImageZono
            
            input_image : Image -> the input ImageStar or ImageZono
            
            returns a zero-padded image
        """
    
        if np.sum(self.attributes[MAXP2D_PADDING_SIZE_ID]) == 0:
            return input_image
        else:
            new_c = self.get_zero_padding_input(input_image.get_V()[:, :, :, 0])
            k = new_c.shape
            
            v_shape = input_image.get_V().shape()
            
            new_V = np.zeros((v_shape[0], v_shape[1], v_shape[2], v_shape[3] + 1))
            
            for i in range(input_image.get_pred_num()):
                new_V[:, :, :, i + 1] = self.get_zero_padding_input(input_image.get_V()[:, :, :, i])
                
            new_V[:, :, :, 0] = new_c
            
            if not self.isempty(input_image.get_im_lb()):
                new_im_lb = self.get_zero_padding_input(input_image.get_im_lb())
                new_im_ub = self.get_zero_padding_input(input_image.get_im_ub())
            else:
                new_im_lb = np.array([])
                new_im_ub = np.array([])
                
            if isinstance(input_image, ImageStar):
                return ImageStar(new_V, input_image.get_C(), input_image.get_d(), \
                                 input_image.get_pred_lb(),input_image.get_pred_ub(), \
                                 new_im_lb, new_im_ub)
    
    def step_split_single_input(self, *args):
        """
            Performs a spetSplit operation for a single image.
            A single images can be split into several images in
            a single exact max pooling operation
            
            input_image : np.array([*]) -> the current maxMap of the input image
            ori_image : np.array([*]) -> the original image to compute the maxMap
            pos : np.array([]) -> local position of the maxMap
            split_index : np.array([*]) -> the split indices of the loal pixels where splitting occured
            NOT IMPLEMENTED YET => option (optional) : string -> 'single' - single process computations
                                                                 'parallel' - parallel processes computations
             
            returns a split image                                                   
        """
    
        assert isinstance(input_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
               
        assert isinstance(ori_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
    
        split_index_size = split_index.shape
        
        assert split_index_size[1] == 3 or split_index_size[0] >= 1, 'error: %s' % MAXP2D_ERRMSG_INVALID_SPLIT_INDEX
    
        images = []
        
        for i in range(split_index_size[0]):
            center = split_indes[i, :, :]
            
            other_indices = split_index
            other_indices[i, :, :] = np.array([])
            
            new_C, new_d = ImageStar.is_max(input_image, ori_image, center, other_indices)
            
            if not self.isempty(new_C) and not self.isempty(new_d):
                new_V = input_image.get_V()
                new_V[pos[0], pos[1], pos[2], :] = ori_image.get_V()[center[0], center[1], center[2], :]
                
                image = ImageStar(new_V, new_C, new_d, input_image.get_pred_lb(), input_image.get_pred_ub(), \
                                  input_image.get_im_lb(), input_image.get_im_ub())
                
                image.set_max_indices(input_image.get_max_indices())
                
                image.set_input_sizes(input_image.get_input_sizes())
                
                image.update_max_idx(self.attributes[MAXP2D_ARGS_NAME_ID], center, pos)
                
                images.append(image)
                
        return images
    
    def step_split_multiple_inputs(self, *args):
        """
            Performs a spetSplit operation for multiple images.
            A single images can be split into several images in
            a single exact max pooling operation
            
            input_image : np.array([*]) -> the current maxMap of the input image
            ori_image : np.array([*]) -> the original image to compute the maxMap
            pos : np.array([]) -> local position of the maxMap
            split_index : np.array([*]) -> the split indices of the loal pixels where splitting occured
            NOT IMPLEMENTED YET => option (optional) : string -> 'single' - single process computations
                                                                 'parallel' - parallel processes computations
             
            returns split images                                                    
        """
            
        images = []
    
        for i in range (len(args[MAXP2D_STEP_SPLIT_ARGS_INPUT_IMGS_ID])):
            images.append(self.step_split(args[MAXP2D_STEP_SPLIT_ARGS_INPUT_IMGS_ID][i], \
                                          args[MAXP2D_STEP_SPLIT_ARGS_ORI_IMG_ID], \
                                          args[MAXP2D_STEP_SPLIT_ARGS_POS_ID], \
                                          args[MAXP2D_STEP_SPLIT_ARGS_SPLIT_INDEX_ID]))
    
    
        return images
    
    def reach_approx_single_input(self, *args):
        """
            Performs an over-approximate reachability analysis on the given image
            
            input_image : ImageStar -> the input ImageStar set
            
            returns the over-approximation of the exact reachability set for the given imagestar
        """
    
        input_image = args[MAXP2D_RA_ARGS_INPUT_IMG_ID]
    
        assert isinstance(input_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
    
        h, w = self.get_size_maxMap(input_image.get_V()[:, :, 0, 0])
        start_points = self.get_start_points(input_image.get_V()[:, :, 0, 0])
        
        max_index = np.empty((h, w, input_image.get_num_channel()))
    
        padded_image = self.get_zero_padding_imageStar(input_image)
        
        np = padded_image.get_num_pred()
        counter = 0
        
        for k in range(padded_image.get_num_channel):
            for i in range(h):
                for j in range(w):
                    max_index[i, j, k] = padded_image.get_local_max_index(start_points[i, j], \
                                                                          self.attributes[MAXP2D_POOL_SIZE_ID], \
                                                                          k)
                    max_id = max_index[i, j, k]
                    
                    if max_id.shape[0] > 1:
                        np + 1
                        counter += 1
    
        new_V = np.zeros[h, w, padded_image.get_num_channel() + 1, np + 1]
        
        new_pred_id = 0
        
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i, j, k]
                    
                    if max_id.shape[0] == 1:
                        for p in range(pad_image.get_num_pred() + 1):
                            new_V[i, j, k, p] = padded_image.get_V()[max_id[0], max_id[1], k, p]
                    else:
                        new_V[i, j, k, 1] = 0
                        new_pred_id += 1
                        new_V[i, j, k, pad_image.get_num_pred() + 1 + new_pred_id] = 1
    
        total_pool_size = np.prod(self.attributes[MAXP2D_POOL_SIZE_ID])
        
        new_C = np.zeros((new_pred_id * (total_pool_size + 1), np))
        new_d = np.zeros((new_pred_id * (total_pool_size + 1), 1))
        new_pred_lb = np.zeros((new_pred_id, 1))
        new_pred_ub = np.zeros((new_pred_id, 1))
        
        new_pred_id = 0
    
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i, j, k]
                    
                    if max_id.shape[0] > 1:
                        # first constraint
                        new_pred_index = new_pred_index + 1
                        start_point = start_points[i,j]
                        
                        local_points = padded_image.get_local_points(start_points, self.attributes[MAXP2D_POOL_SIZE_ID])
                        
                        C1 = np.zeros((1, np))
                        C1[pad_image.get_num_pred() + new_pred_index] = 1
                        
                        lb, ub = padded_image.get_local_bounds(start_points, self.attributes[MAXP2D_POOL_SIZE_ID], k)
                        
                        new_pred_lb[new_pred_index] = lb
                        new_pred_ub[new_pred_index] = ub
                        
                        d = ub
                    
                        # second constraint
                        C2 = np.zeros((total_pool_size, np))
                        d2 = np.zeros((total_pool_size, 1))
                        
                        for g in range(total_pool_size):
                            point = local_points[g, :]
                            
                            C2[g, 1:padded_image.get_num_pred()] = padded_image.get_V()[point[0], point[1], k, 1 : padded_image.get_num_pred() + 1]
                            C2[g, 1:padded_image.get_num_pred() + new_pred_inex] = -1
                            d2[g] = -padded_image.get_V()[points[0], points[1], k, 0]
                            
                        C = np.vstack((C1, C2))
                        d = np.vstack((d1, d2))
                        
                        new_C[(new_pred_index - 1) * (total_pool_size + 1) + 1 : new_pred_index * (total_pool_size + 1), :] = C
                        new_d[(new_pred_index - 1) * (total_pool_size + 1) + 1 : new_pred_index * (total_pool_size + 1), :] = d
                        
        C = np.hstack((padded_image.get_C(), np.zeros((padded_image.get_C().shape[0], new_pred_index))))
        
        new_C = np.vstack((C, new_C))
        new_d = np.vstack((padded_image.get_d(), new_d))
        new_pred_lb = np.vstack((padded_image.get_pred_lb(), new_pred_lb))
        new_pred_ub = np.vstack((padded_image.get_pred_ub(), new_pred_ub))
        
        image = ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        image.add_input_size(self.attributes[MAXP2D_NAME_ID], np.hstack((padded_image.get_height(), padded_image.get_width())))
        image.add_max_idx(self.attributes[MAXP2D_NAME_ID], max_index)

        return image
    
    
    def reach_star_approx_multiple_inputs(self, *args):
        """
            Performs an over-approximate reachability analysis on the given images
            
            input_image : ImageStar* -> the input ImageStar sets
            
            returns the over-approximation of the exact reachability sets for the given ImageStar-s
        """
        
        rs_outputs = []
        
        for i in range(len(args[MAXP2D_RA_MULT_INPUT_ID])):
            rs_outputs.append(self.reach_approx_single_input(args[MAXP2D_RA_MULT_INPUT_ID]))
    
        return rs_outputs
    
    def reach_zono_single_input(self, input_image):
        """
            Performs reachability analysis on the given ImageZono
            
            input_image : ImageZono -> the input ImageZono
            
            returns the reachable set of the given ImageZono after applying max pooling
        """
        
        assert isinstance(input_image, ImageZono), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGZ_INPUT
        
        lb = input_image.get_lb_image()
        ub = input_image.get_ub_image()
        
        maxpool = nn.MaxPool2d(kernel_size = self.attributes[MAXP2D_POOL_SIZE_ID],\
                               stride = self.attributes[MAXP2D_STRIDE_ID],\
                               padding = self.attributes[MAXP2D_PADDING_SIZE_ID])
        
        new_lb = maxpool(-lb)
        new_ub = maxpool(ub)
        
        return ImageZono(-new_lb, new_ub)
    
    def reach_zono_multiple_inputs(self, *args):
        """
            Performs reachability analysis on the given set of ImageZono-s
            
            input_image : ImageZono* -> the input images
            
            returns the reachable sets of the given ImageZono-s after applying max pooling
        """
        
        rs_outputs = []
        
        for i in range(len(args[MAXP2D_RZ_MULT_INPUT_ID])):
            rs_outputs.append(self.reach_zono_single_input(args[MAXP2D_RZ_MULT_INPUT_ID]))
    
        return rs_outputs

    def reach(self, *args):
        """
            Performs reachability analysis on the multiple inputs
            
            in_image -> the input image(s)
            method : string -> 'exact-star' - exact star reachability
                            -> 'approx-star' - approx star reachability
                            -> 'approx-zono' - approx zono reachability
                         
            option : int -> 0 - single core
                         -> 1 - multiple cores
            
            returns the output set(s)
        """
                
        method = args[MAXP2D_REACH_ARGS_METHOD_ID]
        option = args[MAXP2D_REACH_ARGS_OPTION_ID]
        
        if method == 'approx-star' or method == 'approx-star':
            IS = self.reach_star_approx_multiple_inputs(args[MAXP2D_REACH_ARGS_INPUT_IMAGES_ID], option)
        elif method == 'exact-star':
            IS = self.reach_star_exact_multiple_inputs(args[MAXP2D_REACH_ARGS_INPUT_IMAGES_ID], option)
        elif method == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[MAXP2D_REACH_ARGS_INPUT_IMAGES_ID], option)
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result