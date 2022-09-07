import numpy as np
import torch
import torch.nn as nn 
import sys
import copy

sys.path.insert(0, "engine/set/imagestar")
from imagestar import *

sys.path.insert(0, "engine/set/imagezono")
from imagezono import *

MAXP2D_ERRMSG_PARAM_NOT_NP = "One of the numerical input parameters is not a numpy arrays"
MAXP2D_ERRMSG_INVALID_POOL_SIZE = 'Invalid pool size'
MAXP2D_ERRMSG_INVALID_STRIDE = 'Invalid stride matrix'
MAXP2D_ERRMSG_INVALID_PADDING_SIZE = 'Invalide padding size'
MAXP2D_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs (should be 0, 3, or 4)'
MAXP2D_ERRMSG_UNK_REACH_METHOD = "Unknown reachability method"
MAXP2D_ERRMSG_INVALID_PRECISION_OPT = 'The given precision option is not supported. Possible options: \'single\', \'double\', \'empty\''
MAXP2D_ERRMSG_EVAL_INVALID_PARAM_NUM = 'Invalid number of input parameters'
MAXP2D_ERRMSG_NAME_NOT_STRING = 'Layer name is not a string'

MAXP2D_ERRMSG_RE_SINGLE_INVALID_ARGS_NUM = 'Invalid numer of arguments to perform reachability analysis'

MAXP2D_ERRORMSG_INVALID_IMGS_INPUT = 'The input image should be an ImageStar'
MAXP2D_ERRORMSG_INVALID_IMGZ_INPUT = 'The input image should be an ImageZono'

MAXP2D_ERRMSG_INVALID_SPLIT_INDEX = 'Invalid split index, it should have 3 columns and at least 1 row'


8 = 8

8 = 8
4 = 4
3 = 3

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
    
    def __init__(self, *args):
            
        if len(args) <= 8 and len(args) > 0:
            if len(args)== 8 or len(args) == 4:
                if len(args)== 8:
                    self.num_inputs = args[4]
                    self.num_outputs = args[6]
                    self.input_names = args[5]
                    self.output_names = args[7]
                
                assert isinstance(args[0], str), 'error: %s' % MAXP2D_ERRMSG_NAME_NOT_STRING
                self.name = args[0]

            if len(args) == 3:
                self.name = MAXP2D_DEFAULT_LAYER_NAME                
                
            if self.isempty(args[1]):
                self.pool_size = MAXP2D_DEFAULT_POOL_SIZE
                self.stride =  MAXP2D_DEFAULT_STRIDE
                self.padding_size = MAXP2D_DEFAULT_PADDING_SIZE
            else:
                
                # TODO: preprocess shape
                assert isinstance(args[1], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert args[1].shape[0] == 1 and \
                       args[1].shape[1] == 2, \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_POOL_SIZE
                    
                assert isinstance(args[2], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert args[2].shape[0] == 1 and \
                       args[2].shape[1] == 2, \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_STRIDE
                            
                assert isinstance(args[3], np.ndarray), 'error: %s' % MAXP2D_ERRMSG_PARAM_NOT_NP
                assert (args[3].shape[0] == 1 and \
                       args[3].shape[1] == 2) or \
                       (args[3].shape[0] == 1 and \
                       args[3].shape[1] == 4), \
                        'error: %s' % MAXP2D_ERRMSG_INVALID_PADDING_SIZE
                            
                self.pool_size = [args[1].astype('int')[0][i] for i in range(args[1].shape[1])]
                self.stride = [args[2].astype('int')[0][i] for i in range(args[2].shape[1])]
                self.padding_size = [args[3].astype('int')[0][i] for i in range(args[3].shape[1])]
        elif len(args) == 0:
                self.name = MAXP2D_DEFAULT_LAYER_NAME                

                self.pool_size = MAXP2D_DEFAULT_POOL_SIZE
                self.stride = MAXP2D_DEFAULT_STRIDE
                self.padding_size = MAXP2D_DEFAULT_PADDING_SIZE

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
        
        if len(args) == 2:
            assert args[1] == 'single' or \
                   args[1] == 'double' or \
                   args[1] == 'empty', \
                   'error: %s' % MAXP2D_ERRMSG_INVALID_PRECISION_OPT
        elif len(args) != 1:
            raise(MAXP2D_ERRMSG_EVAL_INVALID_PARAM_NUM)
        
        # TODO: create two padding variables -> one for evaluation, one for reachability analysis
        current_padding = self.padding_size
        if len(current_padding) == 4 and np.all(current_padding == current_padding[0]):
            current_padding = [current_padding[0], current_padding[0]]
        
        maxpool = nn.MaxPool2d(kernel_size=self.pool_size, \
                             stride=self.stride, \
                             padding=current_padding)
        
        input = args[0]
        if not isinstance(input, torch.FloatTensor):
            input = torch.FloatTensor(input)
            
        return maxpool(input).cpu().detach().numpy()
            
    
    def get_start_points(self, input_image):
        """
            Computes a collection of start points for maxMap
            
            input : np.array([*])
            
            returns a set of start points of maxMap
        """
    
        I = self.get_zero_padding_input(input_image)
        
        h, w = self.get_size_maxMap(input_image, I)
        
        #start_points = [[np.array([0, 0])] * w] * h
        
        start_points = [[np.array([0, 0]) for j in range (w)] for i in range(h)]
        
        for i in range(h):
            for j in range(w):
                if i == 0:
                    start_points[i][j][0] = 0
                if j == 0:
                    start_points[i][j][1] = 0
                    
                if i > 0:
                    start_points[i][j][0] = start_points[i - 1][j][0] + self.stride[0]
                if j > 0:
                    start_points[i][j][1] = start_points[i][j - 1][1] + self.stride[1]
            
                    
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
            padded_image = self.get_zero_padding_input(input_image)
            
        input_size = padded_image.shape
        
        h = np.floor((input_size[0] - self.stride[0]) / self.stride[0] + 1)
        w = np.floor((input_size[1] - self.stride[1]) / self.stride[1] + 1)
    
        return int(h), int(w)
    
    def get_zero_padding_input(self, input_image):
        """
            Applies zero padding to the image
            
            input : Image -> the input ImageStar or ImageZono
            
            returns the  padded image
        """
    
        input_shape = input_image.shape
        
        paired_padding = np.array(self.padding_size)
        
        if (len(paired_padding.shape) == 2 or len(paired_padding.shape) == 1):
            paired_padding = np.append(paired_padding, self.padding_size)
        
        t = paired_padding[0]
        b = paired_padding[1]
        l = paired_padding[2]
        r = paired_padding[3]
    
        if len(input_shape) == 2:
            h = input_shape[0]
            w = input_shape[1]
            
            padded_image = np.zeros((t + h + b, l + w + r))
            padded_image[t : t + h, l : l + w] = input_image
            
            return padded_image
        elif len(input_shape) > 2:
            h = input_shape[0]
            w = input_shape[1]
            d = input_shape[2]
            
            padded_image = np.zeros((t + h + b, l + w + r, d))
    
            for i in range(d):
                padded_image[t: t + h, l : l + w, i] = input_image[:, :, i]
                
            return padded_image
        else:
            raise Exception(MAXP2D_ERRMSG_ZERO_PAD_INVALID_INPUT)
        
    def get_zero_padding_imageStar(self, input_image):
        """
            Computes a zero-padded ImageStar or ImageZono
            
            input_image : Image -> the input ImageStar or ImageZono
            
            returns a zero-padded image
        """
    
        if np.sum(self.padding_size) == 0:
            return input_image
        else:
            new_c = self.get_zero_padding_input(input_image.get_V()[:, :, :, 0])
            k = new_c.shape
            
            new_V = np.zeros((np.append(k, input_image.get_num_pred() + 1)))
            
            for i in range(input_image.get_num_pred()):
                new_V[:, :, :, i + 1] = self.get_zero_padding_input(input_image.get_V()[:, :, :, i + 1])
                
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
    
        assert isinstance(args[0], ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
        input_image = args[0]
               
        assert isinstance(args[1], ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
        ori_image = args[1]
    
        split_index = args[3]
        split_index_size = split_index.shape    
        assert split_index_size[1] == 3 or split_index_size[0] >= 1, 'error: %s' % MAXP2D_ERRMSG_INVALID_SPLIT_INDEX
    
        pos = args[2]
    
        images = []
        
        for i in range(split_index_size[0]):
            center = split_index[i]
            
            other_indices = np.copy(split_index)
            other_indices = np.delete(other_indices, i, 0) 
                        
            new_C, new_d = ImageStar.is_max(input_image, ori_image, center, other_indices)
            
            if not self.isempty(new_C) and not self.isempty(new_d):
                new_V = input_image.get_V()
                new_V[pos[0], pos[1], pos[2], :] = ori_image.get_V()[center[0], center[1], center[2], :]
                
                image = ImageStar(new_V, new_C, new_d, input_image.get_pred_lb(), input_image.get_pred_ub(), \
                                  input_image.get_im_lb(), input_image.get_im_ub())
                
                image.set_max_indices(input_image.get_max_indices())
                
                image.set_input_sizes(input_image.get_input_sizes())
                
                image.update_max_idx(self.attributes[0], center, pos)
                
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
    
        if isinstance(args[0], ImageStar):
            images = images + self.step_split_single_input(args[0], \
                                          args[1], \
                                          args[2], \
                                          args[3])
        else:
            for i in range (len(args[0])):
                images = images + self.step_split_single_input(args[0][i], \
                                              args[1], \
                                              args[2], \
                                              args[3])
    
    
        return images
    
    def reach_star_exact_multiple_inputs(self, *args):
        """
            Performs exact reachability analysis on multiple input images
            
            input_images : Image* -> the input images
            dis_opt : string -> display option
            
            returns the reachable sets for the given images
        """
        
        rs = []
        
        input_images = args[0]
        
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
    
        assert len(args) <= 2, 'error: %s' % MAXP2D_ERRMSG_RE_SINGLE_INVALID_ARGS_NUM
        
        input_image = args[0]
        assert isinstance(input_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT    
               
        start_points = self.get_start_points(input_image.get_V()[:, :, 0, 0])
        h, w = self.get_size_maxMap(input_image.get_V()[:, :, 0, 0])
        
        padded_image = self.get_zero_padding_imageStar(input_image)
        
        #max_index = numpy.empty((h, w, padded_image.get_num_channel())) 
        
        max_index = [[[np.array([-1, -1, -1]) for k in range(padded_image.get_num_channel())] for j in range (w)] for i in range(h)]

        
        max_index_result = copy.deepcopy(max_index)
        
        maxMap_basis_V = np.zeros((h, w, padded_image.get_num_channel(), padded_image.get_num_pred() + 1))
        
        split_pos = np.zeros((1,3))
        
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_index[i][j][k] = padded_image.get_localMax_index(start_points[i][j], \
                                                                        self.attributes[1], \
                                                                        k)
                    
                    if max_index[i][j][k].shape[0] == 1:
                        current_id = max_index[i][j][k][0]
                        maxMap_basis_V[i, j, k, :] = padded_image.get_V()[current_id[0], current_id[1], k, :]
                        max_index_result[i][j][k] = max_index[i][j][k]
                    else:
                        split_pos = np.vstack((split_pos, np.array([i,j,k])))
                        print(split_pos)
          
        split_pos = split_pos[1:split_pos.shape[0], :].astype('int')
                        
        current_split_size = split_pos.shape[0]
        
        #if dis_opt == DEFAULT_DISPLAY_OPTION:
            #print(MAXP2D_MSG_SPLITS_OCCURRED)
            
        images = ImageStar(maxMap_basis_V, padded_image.get_C(), padded_image.get_d(), padded_image.get_pred_lb(),padded_image.get_pred_ub())
        images.add_max_idx(self.name, max_index_result)
        images.add_input_size(self.name, np.array([padded_image.get_height(), padded_image.get_width()]))
        
        if current_split_size > 0:
            for i in range(current_split_size):                    
                images = self.step_split_multiple_inputs(images, padded_image, split_pos[i], max_index[split_pos[i][0]][split_pos[i][1]][split_pos[i][2]], [])
                
                images_num_post_split = len(images)
                
                #if dis_opt == DEFAULT_DISPLAY_OPTION:
                    #print(MAXP2D_MSG_SPLIT_DETAILS)
                    
        return images
    
    

    
    def reach_approx_single_input(self, *args):
        """
            Performs an over-approximate reachability analysis on the given image
            
            input_image : ImageStar -> the input ImageStar set
            
            returns the over-approximation of the exact reachability set for the given imagestar
        """
    
        input_image = args[0]
    
        assert isinstance(input_image, ImageStar), 'error: %s' % MAXP2D_ERRORMSG_INVALID_IMGS_INPUT
    
        h, w = self.get_size_maxMap(input_image.get_V()[:, :, 0, 0])
        start_points = self.get_start_points(input_image.get_V()[:, :, 0, 0])
            
        padded_image = self.get_zero_padding_imageStar(input_image)
        
        max_index = [[[np.array([]) for k in range(padded_image.get_num_channel())] for j in range (w)] for i in range(h)]
        
        num_p = padded_image.get_num_pred()
        counter = 0
        
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_index[i][j][k] = padded_image.get_localMax_index(start_points[i][j], \
                                                                        self.attributes[1], \
                                                                        k)
                    max_id = max_index[i][j][k]
                    
                    if max_id.shape[0] > 1:
                        num_p += 1
                        counter += 1
    
        new_V = np.zeros((h, w, padded_image.get_num_channel(), num_p + 1))
        
        new_pred_id = 0
        
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    
                    if max_id.shape[0] == 1:
                        for p in range(padded_image.get_num_pred() + 1):
                            new_V[i, j, k, p] = padded_image.get_V()[max_id[0][0], max_id[0][1], k, p]
                    else:
                        new_V[i, j, k, 0] = 0
                        new_pred_id += 1
                        new_V[i, j, k, padded_image.get_num_pred() + new_pred_id] = 1
    
        total_pool_size = np.prod(self.pool_size)
        
        new_C = np.zeros((new_pred_id * (total_pool_size + 1), num_p))
        new_d = np.zeros((new_pred_id * (total_pool_size + 1), 1))
        new_pred_lb = np.zeros((new_pred_id, 1))
        new_pred_ub = np.zeros((new_pred_id, 1))
        
        new_pred_index = 0
    
        for k in range(padded_image.get_num_channel()):
            for i in range(h):
                for j in range(w):
                    max_id = max_index[i][j][k]
                    
                    if max_id.shape[0] > 1:
                        # first constraint
                        new_pred_index = new_pred_index + 1
                        start_point = start_points[i][j]
                        
                        local_points = padded_image.get_local_points(start_point, self.pool_size).astype('int')
                        
                        C1 = np.zeros((1, num_p))
                        C1[0,padded_image.get_num_pred() + new_pred_index - 1] = 1
                        
                        lb, ub = padded_image.get_local_bound(start_point, self.pool_size, k)
                        
                        new_pred_lb[new_pred_index - 1] = lb
                        new_pred_ub[new_pred_index - 1] = ub
                        
                        d1 = ub
                    
                        # second constraint
                        C2 = np.zeros((total_pool_size, num_p))
                        d2 = np.zeros((total_pool_size, 1))
                        
                        for g in range(total_pool_size):
                            point = local_points[g]
                            
                            C2[g, 0:padded_image.get_num_pred()] = padded_image.get_V()[point[0], point[1], k, 1 : padded_image.get_num_pred() + 1]
                            C2[g, padded_image.get_num_pred() + new_pred_index - 1] = -1
                            d2[g] = -padded_image.get_V()[point[0], point[1], k, 0]
                            
                        C = np.vstack((C1, C2))
                        d = np.vstack((d1, d2))
                        
                        new_C[(new_pred_index - 1) * (total_pool_size + 1) : new_pred_index * (total_pool_size + 1), :] = C
                        new_d[(new_pred_index - 1) * (total_pool_size + 1) : new_pred_index * (total_pool_size + 1), :] = d
                        
        C = np.hstack((padded_image.get_C(), np.zeros((padded_image.get_C().shape[0], new_pred_index))))
        
        new_C = np.vstack((C, new_C))
        new_d = np.vstack((padded_image.get_d(), new_d))
        
        bound_size = padded_image.get_pred_lb().shape[0]
        
        # TODO: design so that all the arrays would have both dimensions
        padded_im_lb = padded_image.get_pred_lb()
        padded_im_ub = padded_image.get_pred_ub()
        
        if len(padded_im_lb.shape) == 1:
            padded_im_lb = np.reshape(padded_im_lb, (padded_im_lb.shape[0], 1))
            padded_im_ub = np.reshape(padded_im_ub, (padded_im_ub.shape[0], 1))
        
        new_pred_lb = np.vstack((padded_im_lb, new_pred_lb))
        new_pred_ub = np.vstack((padded_im_ub, new_pred_ub))
        
        image = ImageStar(new_V, new_C, new_d, new_pred_lb, new_pred_ub)
        image.add_input_size(self.name, np.hstack((padded_image.get_height(), padded_image.get_width())))
        image.add_max_idx(self.name, max_index)

        return image
    
    
    def reach_star_approx_multiple_inputs(self, *args):
        """
            Performs an over-approximate reachability analysis on the given images
            
            input_image : ImageStar* -> the input ImageStar sets
            
            returns the over-approximation of the exact reachability sets for the given ImageStar-s
        """
        
        rs_outputs = []
        
        for i in range(len(args[0])):
            rs_outputs.append(self.reach_approx_single_input(args[0][i]))
    
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
        
        maxpool = nn.MaxPool2d(kernel_size = self.pool_size,\
                               stride = self.stride,\
                               padding = self.padding_size)
        
        new_lb = maxpool(torch.FloatTensor(-lb)).cpu().detach().numpy()
        new_ub = maxpool(torch.FloatTensor(ub)).cpu().detach().numpy()
        
        return ImageZono(-new_lb, new_ub)
    
    def reach_zono_multiple_inputs(self, *args):
        """
            Performs reachability analysis on the given set of ImageZono-s
            
            input_image : ImageZono* -> the input images
            
            returns the reachable sets of the given ImageZono-s after applying max pooling
        """
        
        rs_outputs = []
        
        for i in range(len(args[0])):
            rs_outputs.append(self.reach_zono_single_input(args[0][i]))
    
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
                
        method = args[1]
        option = args[2]
        
        if method == 'approx-star':
            IS = self.reach_star_approx_multiple_inputs(args[0], option)
        elif method == 'exact-star':
            IS = self.reach_star_exact_multiple_inputs(args[0], option)
        elif method == 'approx-zono':
            IS = self.reach_zono_multiple_inputs(args[0], option)
        
########################## UTILS ##########################
    def offset_args(self, args, offset):
        result = []
            
        for i in range(len(args) + offset):
            result.append(np.array([]))
                
            if i >= offset:
                result[i] = args[i - offset]
                    
        return result
    
    def isempty(self, param):
        if not isinstance(param, np.ndarray):
            return param == []
        else:
            return param.size == 0 or (param is np.array and param.shape[0] == 0)
    
    