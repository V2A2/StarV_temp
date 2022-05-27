#!/usr/bin/python3
import numpy as np
import gurobipy as gp
from gurobipy import GRB 

import sys
import operator
        
sys.path.insert(0, "../../../engine/set/")
        
from star import *

############################ ERROR MESSAGES ############################

#INVALID_PARAM_NUMBER_MSG = "ImageStar does not support this number of parameters"

ERRMSG_INCONSISTENT_CONSTR_DIM = "Inconsistent dimension between constraint matrix and constraint vector"
ERRMSG_INCONSISTENT_PRED_BOUND_DIM = "Number of predicates is different from the size of the lower bound or upper bound predicate vector"
ERRMSG_INCONSISTENT_BOUND_DIM = "Invalid lower/upper bound predicate vector, vector should have one column"
ERRMSG_INVALID_CONSTR_VEC = "Invalid constraint vector, vector should have one column"
ERRMSG_INVALID_BASE_MATRIX = "Invalid basis matrix"
ERRMSG_INCONSISTENT_BASIS_MATRIX_PRED_NUM = "Inconsistency between the basis matrix and the number of predicate variables"

ERRMSG_INCONSISTENT_LB_DIM = "Inconsistent dimension between lower bound image and the constructed imagestar"
ERRMSG_INCONSISTENT_UB_DIM = "Inconsistent dimension between upper bound image and the constructed imagestar"

ERRMSG_INCONSISTENT_CENTER_IMG_ATTACK_MATRIX = "Inconsistency between center image and attack bound matrices"
ERRMSG_INCONSISTENT_CHANNELS_NUM = "Inconsistent number of channels between the center image and the bound matrices"

ERRMSG_INCONSISTENT_LB_UB_DIM = "Inconsistency between lower bound image and upper bound image"

ERRMSG_INVALID_INIT = "Invalid number of input arguments, (should be from 0, 3, 5 , or 7)"

ERRMSG_IMGSTAR_EMPTY = "The ImageStar is empty"

ERRMSG_INVALID_PREDICATE_VEC = "Invalid predicate vector"

ERRMSG_INCONSISTENT_PREDVEC_PREDNUM = "Inconsistency between the size of the predicate vector and the number of predicates in the imagestar"

ERRMSG_INCONSISTENT_SCALE_CHANNELS_NUM = "Inconsistent number of channels between scale array and the ImageStar"

ERRMSG_INCONSISTENT_IMGDIM_IMGSTAR = "Inconsistent dimenion between input image and the ImageStar"

ERRMSG_INVALID_INPUT_IMG = "Invalid input image"

ERRMSG_INVALID_INPUT_POINT = "Invalid input point"
ERRMSG_INVALID_FIRST_INPUT_POINT = "The first input point is invalid"
ERRMSG_INVALID_SECOND_INPUT_POINT = "The second input point is invalid"

ERRMSG_INVALID_VERT_ID = "Invalid veritical index"
ERRMSG_INVALID_HORIZ_ID = "Invalid horizonal index"
ERRMSG_INVALID_CHANNEL_ID = "Invalid channel index"



ESTIMATE_RANGE_STAGE_STARTED = "Ranges estimation started..."
ESTIMATE_RANGE_STAGE_OVER = "Ranges estimation finished..."

DISPLAY_ON_OPTION = "disp"
############################ PARAMETERS IDS ############################

##### ATTRIBUTES:
V_ID = 0
C_ID = 1
D_ID = 2
PREDLB_ID = 3
PREDUB_ID = 4
IM_LB_ID = 5
IM_UB_ID = 6
IM_ID = 7
LB_ID = 8
UB_ID = 9

NUMPRED_ID = 10
HEIGHT_ID = 11
WIDTH_ID = 12
NUM_CHANNEL_ID = 13
FLATTEN_ORDER_ID = 14

LAST_ATTRIBUTE_ID = FLATTEN_ORDER_ID

##### ARGUMENTS:
VERT_ID = 0
HORIZ_ID = 1
CHANNEL_ID = 2

POINTS_ID = 0

START_POINT_ID = 0
POOL_SIZE_ID = 1

P1_ID = 0
P2_ID = 1
#####################

############################ PARAMETERS NUMBERS ############################
IMAGESTAR_ATTRIBUTES_NUM = LAST_ATTRIBUTE_ID + 1
PREDICATE_IMGBOUNDS_INIT_ARGS_NUM = 7
PREDICATE_INIT_ARGS_NUM = 5
IMAGE_INIT_ARGS_NUM = 3
BOUNDS_INIT_ARGS_NUM = 2
#####################

IM_OFFSET = 7
IMAGE_INIT_ARGS_OFFSET = IM_ID
BOUNDS_INIT_ARGS_OFFSET = IM_LB_ID

DEFAULT_DISP_OPTION = ""

COLUMN_FLATTEN = 'F'
############################## DEFAULT VALUES ##############################

DEFAULT_SOLVER_ARGS_NUM = 3
CUSTOM_SOLVER_ARGS_NUM = 4


class ImageStar:
    # Class for representing set of images using Star set
    # An image can be attacked by bounded noise. An attacked image can
    # be represented using an ImageStar Set
    # author: Mykhailo Ivashchenko
    # date: 2/14/2022

    #=================================================================%
    #   a 3-channels color image is represented by 3-dimensional array 
    #   Each dimension contains a h x w matrix, h and w is the height
    #   width of the image. h * w = number of pixels in the image.
    #   *** A gray image has only one channel.
    #
    #   Problem: How to represent a disturbed(attacked) image?
    #   
    #   Use a center image (a matrix) + a disturbance matrix (positions
    #   of attacks and bounds of corresponding noises)
    #
    #   For example: Consider a 4 x 4 (16 pixels) gray image 
    #   The image is represented by 4 x 4 matrix:
    #               IM = [1 1 0 1; 0 1 0 0; 1 0 1 0; 0 1 1 1]
    #   This image is attacked at pixel (1,1) (1,2) and (2,4) by bounded
    #   noises:     |n1| <= 0.1, |n2| <= 0.2, |n3| <= 0.05
    #
    #
    #   Lower and upper noises bounds matrices are: 
    #         LB = [-0.1 -0.2 0 0; 0 0 0 -0.05; 0 0 0 0; 0 0 0 0]
    #         UB = [0.1 0.2 0 0; 0 0 0 0.05; 0 0 0 0; 0 0 0 0]
    #   The lower and upper bounds matrices also describe the position of 
    #   attack.
    #
    #   Under attack we have: -0.1 + 1 <= IM(1,1) <= 1 + 0.1
    #                         -0.2 + 1 <= IM(1,2) <= 1 + 0.2
    #                            -0.05 <= IM(2,4) <= 0.05
    #
    #   To represent the attacked image we use IM, LB, UB matrices
    #   For multi-channel image we use multi-dimensional array IM, LB, UB
    #   to represent the attacked image. 
    #   For example, for an attacked color image with 3 channels we have
    #   IM(:, :, 1) = IM1, IM(:,:,2) = IM2, IM(:,:,3) = IM3
    #   LB(:, :, 1) = LB1, LB(:,:,2) = LB2, LB(:,:,3) = LB3
    #   UB(:, :, 1) = UB1, UB(:,:,2) = UB2, UB(:,:,3) = UB3
    #   
    #   The image object is: image = ImageStar(IM, LB, UB)
    #=================================================================

    # 2D representation of an ImageStar
    # ====================================================================
    #                   Definition of Star2D
    # 
    # A 2D star set S is defined by: 
    # S = {x| x = V[0] + a[1]*V[1] + a[2]*V[2] + ... + a[n]*V[n]
    #           = V * b, V = {c V[1] V[2] ... V[n]}, 
    #                    b = [1 a[1] a[2] ... a[n]]^T                                   
    #                    where C*a <= d, constraints on a[i]}
    # where, V[0], V[i] are 2D matrices with the same dimension, i.e., 
    # V[i] \in R^{m x n}
    # V[0] : is called the center matrix and V[i] is called the basic matrix 
    # [a[1]...a[n] are called predicate variables
    # C: is the predicate constraint matrix
    # d: is the predicate constraint vector
    # 
    # The notion of Star2D is more general than the original Star set where
    # the V[0] and V[i] are vectors. 
    # 
    # Dimension of Star2D is the dimension of the center matrix V[0]
    # 
    # ====================================================================


    def __init__(self, *args):
        """
            Constructor using 2D representation / 1D representation of an ImageStar
            
            args : np.array([params]) -> a list of initial arguments
            
            params can inlude =>
            
            ================ First initialization option ================
            V : np.array([]) -> a cell (size = numPred)
            C : np.array([]) -> a constraints matrix of the predicate
            d : np.array([]) -> a constraints vector of the predicate
            pred_lb : np.array([]) -> lower bound vector of the predicate
            pred_ub : np.array([]) -> upper bound vector of the predicate
            im_lb : np.array([]) -> lower bound image of the ImageStar
            im_ub : np.array([]) -> upper bound image of the ImageStar
            =============================================================
            
            ================ Second initialization option ================
            V : np.array([]) -> a cell (size = numPred)
            C : np.array([]) -> a constraints matrix of the predicate
            d : np.array([]) -> a constraints vector of the predicate
            pred_lb : np.array([]) -> lower bound vector of the predicate
            pred_ub : np.array([]) -> upper bound vector of the predicate
            =============================================================
            
            ================ Third initialization option ================
            IM : np.array([]) -> center image (high-dimensional array)
            LB : np.array([]) -> lower bound of attack (high-dimensional array)
            UB : np.array([]) -> upper bound of attack (high-dimensional array
            =============================================================
        """
        
        self.validate_params(args)
        
        self.attributes = []       
        
        for i in range(IMAGESTAR_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
        
        self.scalar_attributes_ids = [
                NUMPRED_ID, HEIGHT_ID, WIDTH_ID, NUM_CHANNEL_ID
            ]
        
        self.attributes[FLATTEN_ORDER_ID] = COLUMN_FLATTEN
        
        if len(args) == PREDICATE_IMGBOUNDS_INIT_ARGS_NUM or len(args) == PREDICATE_INIT_ARGS_NUM:    
            if np.size(args[V_ID]) and np.size(args[C_ID]) and np.size(args[D_ID]) and np.size(args[PREDLB_ID]) and np.size(args[PREDUB_ID]):
                assert (args[C_ID].shape[0] == 1 and np.size(args[D_ID]) == 1) or (args[C_ID].shape[0] == args[D_ID].shape[0]), \
                       'error: %s' % ERRMSG_INCONSISTENT_CONSTR_DIM
                
                assert (np.size(args[D_ID]) == 1) or (len(np.shape(args[D_ID])) == 1), 'error: %s' % ERRMSG_INVALID_CONSTR_VEC
                
                self.attributes[NUMPRED_ID] = args[C_ID].shape[1];
                self.attributes[C_ID] = args[C_ID].astype('float64')
                self.attributes[D_ID] = args[D_ID].astype('float64')
                
                assert args[C_ID].shape[1] == args[PREDLB_ID].shape[0] == args[PREDUB_ID].shape[0], 'error: %s' % ERRMSG_INCONSISTENT_PRED_BOUND_DIM
                
                assert len(args[PREDLB_ID].shape) == len(args[PREDUB_ID].shape) == 1 or args[PREDUB_ID].shape[1], 'error: %s' % ERRMSG_INCONSISTENT_BOUND_DIM
                
                self.attributes[PREDLB_ID] = args[PREDLB_ID].astype('float64')
                self.attributes[PREDUB_ID] = args[PREDUB_ID].astype('float64')
                
                n = args[V_ID].shape
                
                if len(n) < 2:
                    raise Exception('error: %s' % ERRMSG_INVALID_BASE_MATRIX)
                else:
                    self.attributes[HEIGHT_ID] = n[0]
                    self.attributes[WIDTH_ID] = n[1]
                    
                    self.attributes[V_ID] = args[V_ID]
                    
                    if len(n) == 4:
                        assert n[3] == self.attributes[NUMPRED_ID] + 1, 'error: %s' % ERRMSG_INCONSISTENT_BASIS_MATRIX_PRED_NUM
                        
                        self.attributes[NUM_CHANNEL_ID] = n[2]
                    else:
                        # TODO: ASK WHY THIS HAPPENS AFTER THE ASSIGNMENT IN LINE 205
                        #self.attributes[NUMPRED_ID] = 0
                        
                        if len(n) == 3:
                            self.attributes[NUM_CHANNEL_ID] = n[2]
                        elif len(n) == 2:
                            self.attributes[NUM_CHANNEL_ID] = 1
                
                if len(args) == PREDICATE_IMGBOUNDS_INIT_ARGS_NUM: 
                    if args[IM_LB_ID].shape[0] != 0 and (args[IM_LB_ID].shape[0] != self.attributes[HEIGHT_ID] or args[IM_LB_ID].shape[1] != self.attributes[WIDTH_ID]):
                        raise Exception('error: %s' % ERRMSG_INCONSISTENT_LB_DIM)
                    else:
                        self.attributes[IM_LB_ID] = args[IM_LB_ID].astype('float64')      
                        
                    if args[IM_UB_ID].shape[0] != 0 and (args[IM_UB_ID].shape[0] != self.attributes[HEIGHT_ID] or args[IM_UB_ID].shape[1] != self.attributes[WIDTH_ID]):
                        raise Exception('error: %s' % ERRMSG_INCONSISTENT_UB_DIM)
                    else:
                        self.attributes[IM_UB_ID] = args[IM_UB_ID].astype('float64')
                    
        elif len(args) == IMAGE_INIT_ARGS_NUM:
            args = self.offset_args(args, IMAGE_INIT_ARGS_OFFSET)
            if np.size(args[IM_ID]) and np.size(args[LB_ID]) and np.size(args[UB_ID]) and args[V_ID].shape[0] == 0:
                n = args[IM_ID].shape
                l = args[LB_ID].shape
                u = args[UB_ID].shape
                
                assert (n[0] == l[0] == u[0] and n[1] == l[1] == u[1]) and (len(n) == len(l) == len(u)), 'error: %s' % ERRMSG_INCONSISTENT_CENTER_IMG_ATTACK_MATRIX
                            
                assert len(n) == len(l) == len(u), 'error: %s' % ERRMSG_INCONSISTENT_CHANNELS_NUM
                
                self.attributes[IM_ID] = args[IM_ID].astype('float64')
                self.attributes[LB_ID] = args[LB_ID].astype('float64')
                self.attributes[UB_ID] = args[UB_ID].astype('float64')
                
                self.attributes[HEIGHT_ID] = n[0]
                self.attributes[WIDTH_ID] = n[1]
                
                if len(n) == 2:
                    self.attributes[NUM_CHANNEL_ID] = 2
                elif len(n) == 3:
                    self.attributes[NUM_CHANNEL_ID] = 3
                else:
                    raise Exception('error: %s' % ERRMSG_INCONSISTENT_CHANNELS_NUM)
                
                self.attributes[IM_LB_ID] = self.attributes[IM_ID] + self.attributes[LB_ID]
                self.attributes[IM_UB_ID] = self.attributes[IM_ID] + self.attributes[UB_ID]
                
                n = self.attributes[IM_LB_ID].shape
                
                I = 0
                
                if len(n) == 3:
                    #TODO: Star returns 'can't create Star set' error because StarV Star constructor initialization does not correspond to the implementation in NNV
                    I = Star(self.attributes[IM_LB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID]), self.attributes[IM_UB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID]))
                    self.attributes[V_ID] = np.reshape(I.V, (I.nVar + 1, n[0] * n[1] * n[2]))
                else:
                    I = Star(self.attributes[IM_LB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID]), self.attributes[IM_UB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID]))
                    self.attributes[V_ID] = np.reshape(I.V, (I.nVar + 1, n[0] * n[1]))
                    
                self.attributes[C_ID] = I.C
                self.attributes[D_ID] = I.d
                
                # TODO: ask why does Star have predicate_lb and ImageStar pred_lb?
                self.attributes[PREDLB_ID]  = I.predicate_lb
                self.attributes[PREDUB_ID] = I.predicate_ub
                
                self.attributes[NUMPRED_ID] = I.nVar
        elif len(args) == BOUNDS_INIT_ARGS_NUM:
            args = self.offset_args(args, BOUNDS_INIT_ARGS_OFFSET)
            if np.size(args[IM_LB_ID]) and np.size(args[IM_UB_ID]) and args[V_ID].shape[0] == 0: #and np.shape(args[IM_ID])[0] == 0:
                lb_shape = args[IM_LB_ID].shape
                ub_shape = args[IM_UB_ID].shape
                
                assert len(lb_shape) == len(ub_shape), 'error: %s' % ERRMSG_INCONSISTENT_LB_UB_DIM
                
                for i in range(len(lb_shape)):
                    assert lb_shape[i] == ub_shape[i], 'error: %s' % ERRMSG_INCONSISTENT_LB_UB_DIM
                    
                lb = args[IM_LB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID])
                ub = args[IM_UB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID])
                
                #TODO: Star returns 'can't create Star set' error because StarV Star constructor initialization does not correspond to the implementation in NNV 
                S = Star(lb, ub)
                    
                self.copy_deep(S.toImageStar)
                    
                self.attributes[IM_LB_ID] = lb_im.astype('float64')
                self.attributes[IM_UB_ID] = im_ub.astype('float64')
        elif self.isempty_init(args):
            self.init_empty_imagestar()
        else:
            raise Exception('error: %s' % ERRMSG_INVALID_INIT)
        
    def sample(self, N):
        """
            Rangomly generates a set of images from an imagestar set
        
            N : int -> number of images
            return -> set of images
        """
        
        assert (not self.isempty(self.attributes[V_ID])), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        if self.isempty(self.attributes[C_ID]) or self.isempty(self.attributes[D_ID]):
            return self.attributes[IM_ID]
        else:
            new_V = np.hstack((np.zeros((self.attributes[NUMPRED_ID], 1)), np.eye(self.attributes[NUMPRED_ID])))
            #TODO: Star returns an error when checking the dimensions even though they match
            S = Star(new_V, self.attributes[C_ID], self.attributes[D_ID])
            pred_samples = S.sample(N)
            
            images = []
            
            for i in range(len(pred_samples)):
                images.append(images, np.array(self.evaluate(pred_samples[:, i])))
                
            return images
        
    def evaluate(self, pred_val):
        """
            Evaluate an ImageStar with specific values of predicates
            
            pred_val : *int -> a vector of predicate variables
            return -> evaluated image
        """            
        
        assert (not self.isempty(self.attributes[V_ID])), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        assert len(pred_val.shape) == 1, 'error: %s' % ERRMSG_INVALID_PREDICATE_VEC
        
        assert pred_val.shape[0] == self.attributes[NUMPRED_ID], 'error: %s' % ERRMSG_INCONSISTENT_PREDVEC_PREDNUM
        
        image = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))
        
        for i in range(self.attributes[NUM_CHANNEL_ID]):
            image[:, :, i] = self.attributes[V_ID][:, :, i, 1]
            
            for j in range(2, self.attributes[NUMPRED_ID] + 1):
                image[:, :, i] = image[:, :, i] + pred_val[j - 1] * self.attributes[V_ID][:, :, i, j]
 
        return image
 
    def affine_map(self, scale, offset):
        """
            Performs affine mapping of the ImageStar: y = scale * x + offset
            
            scale : *float -> scale coefficient [1 x 1 x num_channel] array
            offset : *float -> offset coefficient [1 x 1 x num_channel] array
            return -> a new ImageStar
        """
 
        assert (self.isempty(scale) or self.is_scalar(scale) or len(scale.shape) == self.attributes[NUM_CHANNEL_ID]), 'error: %s' % ERRMSG_INCONSISTENT_SCALE_CHANNELS_NUM
        
        new_V = 0
        
        if not self.isempty(scale):
            new_V = np.multiply(scale, self.attributes[V_ID])
        else:
            new_V = self.attributes[V_ID]
        
        # Affine Mapping changes the center
        if not self.isempty(offset):
            new_V[:, :, :, 0] = new_V[:, :, :, 0] + offset
            
        return ImageStar(new_V, self.attributes[C_ID], self.attributes[D_ID], self.attributes[PREDLB_ID], self.attributes[PREDUB_ID])
    
    def to_star(self):
        """
            Converts current ImageStar to Star
            
            return -> created Star
        """
 
        pixel_num = self.attributes[HEIGHT_ID] * self.attributes[WIDTH_ID] * self.attributes[NUM_CHANNEL_ID]
        
        new_V = np.zeros((pixel_num, self.attributes[NUMPRED_ID] + 1))
        
        if self.isempty(new_V):
            # TODO: error: failed to create Star set
            return Star()
        else:
            for j in range(self.attributes[NUMPRED_ID] + 1):
                #new_V[:, j] = np.reshape(self.attributes[V_ID][:, :, :, j], (pixel_num, 0))
                new_V[:, j] = self.attributes[V_ID][:, :, :, j].flatten(order=self.attributes[FLATTEN_ORDER_ID])
                
            if not self.isempty(self.attributes[IM_LB_ID]) and not self.isempty(self.attributes[IM_UB_ID]):
                state_lb = self.attributes[IM_LB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID])
                state_ub = self.attributes[IM_UB_ID].flatten(order=self.attributes[FLATTEN_ORDER_ID])
                
                # TODO: error: failed to create Star set
                S = Star(new_V, self.attributes[C_ID], self.attributes[D_ID], self.attributes[PREDLB_ID], self.attributes[PREDUB_ID], state_lb, state_ub)
            else:
                # TODO: error: failed to create Star set
                S = Star(new_V, self.attributes[C_ID], self.attributes[D_ID], self.attributes[PREDLB_ID], self.attributes[PREDUB_ID])
                
            return S
 
    def is_empty_set(self):
        """
            Checks if the ImageStar is empty
            
            return -> True if empty, False if isn't empty
        """
 
        S = self.to_star()
        return S.isEmptySet()
    
    def contains(self, image):
        """
            Checks if the ImageStar contains the image
            
            image : *float -> input image
            return -> = 1 if the ImageStar contain the image
                      = 0 if the ImageStar does not contain the image
        """
 
        img_size = image.shape
        
        if len(img_size) == 2: # one channel image
            assert (img_size[0] == self.attributes[HEIGHT_ID] and img_size[1] == self.attributes[WIDTH_ID] and self.attributes[NUM_CHANNEL_ID] == 1), 'error: %s' % ERRMSG_INCONSISTENT_IMGDIM_IMGSTAR
        elif len(img_size) == 3:
            assert (img_size[0] == self.attributes[HEIGHT_ID] and img_size[1] == self.attributes[WIDTH_ID] and img_size[2] == self.attributes[NUM_CHANNEL_ID]), 'error: %s' % ERRMSG_INCONSISTENT_IMGDIM_IMGSTAR
        else:
            raise Exception('error: %s' % ERRMSG_INVALID_INPUT_IMG)
            
        image_vec = image.flatten(order=self.attributes[FLATTEN_ORDER_ID])
        
        # TODO: error: failed to create Star set
        S = self.to_star()
        
        return S.contains(image_vec)
    
    def project2D(self, point1, point2):
        """
            Projects the ImageStar on the give plane
            
            point1 : int -> first dimension index
            point2 : int -> first dimension index
            return -> projected Star
        """
            
        assert (len(point1) == 3 and len(point2) == 3), 'error: %s' % ERRMSG_INVALID_INPUT_POINT
        assert self.validate_point_dim(point1, self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID]), 'error: %s' % ERRMSG_INVALID_FIRST_INPUT_POINT
        assert self.validate_point_dim(point2, self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID]), 'error: %s' % ERRMSG_INVALID_SECOND_INPUT_POINT
        
        point1 -= 1
        point2 -= 1
        
        n = self.attributes[NUMPRED_ID] + 1
        
        new_V = np.zeros((2, n))
        
        for i in range(n):
            new_V[0, i] = self.attributes[V_ID][point1[0], point1[1], point1[2], i]
            new_V[1, i] = self.attributes[V_ID][point2[0], point2[1], point2[2], i]
            
        return Star(new_V, self.attributes[C_ID], self.attributes[D_ID], self.attributes[PREDLB_ID], self.attributes[PREDUB_ID])
        
        
    def get_range(self, *args):
        """
            Gets ranges of a state at specific position using the Gurobi solver
            
            args : np.array([params]) -> multimple parameters that include =>
            vert_id : int -> vertica index
            horiz_id : int -> horizontall index
            channel_id : int -> channel index
            
            
            
            return : np.array([
                        xmin : int -> min of (vert_id, horiz_id, channel_id),
                        xmax : int -> max of (vert_id, horiz_id, channel_id)
                        ])
        """
        
        assert (len(args) == DEFAULT_SOLVER_ARGS_NUM or len(args) == CUSTOM_SOLVER_ARGS_NUM), 'error: %s' % ERRMSG_GETRANGES_INVALID_ARGS_NUM   
        assert (not self.isempty(self.attributes[C_ID]) and not self.isempty(self.attributes[D_ID])), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        # TODO: THIS SHOULD BE ACCOUNTED FOR WHEN THE DATA IS PASSED
        input_args = np.array([
                args[VERT_ID] - 1,
                args[HORIZ_ID] - 1,
                args[CHANNEL_ID] - 1
            ], dtype=int)
        
        input_args = input_args + 1
        
        # TODO: account for potential custom solver identifier
        args = input_args
        
        assert (args[VERT_ID] > -1 and args[VERT_ID] < self.attributes[HEIGHT_ID]), 'error: %s' % ERRMSG_INVALID_VERT_ID
        assert (args[HORIZ_ID] > -1 and args[HORIZ_ID] < self.attributes[WIDTH_ID]), 'error: %s' % ERRMSG_INVALID_HORIZ_ID
        assert (args[CHANNEL_ID] > -1 and args[CHANNEL_ID] < self.attributes[NUM_CHANNEL_ID]), 'error: %s' % ERRMSG_INVALID_CHANNELNUM_ID
        
        bounds = [np.array([]), np.array([])]
        
        f = self.attributes[V_ID][args[VERT_ID], args[HORIZ_ID], args[CHANNEL_ID], 1:self.attributes[NUMPRED_ID] + 1]
        
        if (f == 0).all():
            bounds[XMIN_ID] = self.attributes[V_ID][args[VERT_ID], args[HORIZ_ID], args[CHANNEL_ID], 0]
            bounds[XMAX_ID] = self.attributes[V_ID][args[VERT_ID], args[HORIZ_ID], args[CHANNEL_ID], 0]
        else:
            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            if self.attributes[PREDLB_ID].size and self.attributes[PREDUB_ID].size:
                x = min_.addMVar(shape=self.attributes[NUMPRED_ID], lb=self.attributes[PREDLB_ID], ub=self.attributes[PREDUB_ID])
            else:
                x = min_.addMVar(shape=self.attributes[NUMPRED_ID])
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(self.attributes[C_ID])
            d = np.array(self.attributes[D_ID]).flatten(order=self.attributes[FLATTEN_ORDER_ID])
            min_.addConstr(C @ x <= d)
            min_.optimize()

            if min_.status == 2:
                xmin = min_.objVal + self.attributes[V_ID][args[VERT_ID], args[HORIZ_ID], args[CHANNEL_ID], 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            if self.attributes[PREDLB_ID].size and self.attributes[PREDUB_ID].size:
                x = max_.addMVar(shape=self.attributes[NUMPRED_ID], lb=self.attributes[PREDLB_ID], ub=self.attributes[PREDUB_ID])
            else:
                x = max_.addMVar(shape=self.attributes[NUMPRED_ID])
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(self.attributes[C_ID])
            d = np.array(self.attributes[D_ID]).flatten(order=self.attributes[FLATTEN_ORDER_ID])
            max_.addConstr(C @ x <= d)
            max_.optimize()

            if max_.status == 2:
                xmax = max_.objVal + self.attributes[V_ID][args[VERT_ID], args[HORIZ_ID], args[CHANNEL_ID], 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))

        return np.array([xmin, xmax])

    def estimate_range(self, height_id, width_id, channel_id):
        """
            Estimates a range using only a predicate bounds information
            
            h : int -> height index
            w : int -> width index
            c : int -> channel index
            
            return -> [xmin, xmax]
        """

        assert (not self.isempty(self.attributes[C_ID]) and not self.isempty(self.attributes[D_ID])), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        height_id = int(height_id - 1)
        width_id = int(width_id - 1)
        channel_id = int(channel_id - 1)
        
        assert (height_id > -1 and height_id < self.attributes[HEIGHT_ID]), 'error: %s' % ERRMSG_INVALID_VERT_ID
        assert (width_id > -1 and width_id < self.attributes[WIDTH_ID]), 'error: %s' % ERRMSG_INVALID_HORIZ_ID
        assert (channel_id > -1 and channel_id < self.attributes[NUM_CHANNEL_ID]), 'error: %s' % ERRMSG_INVALID_CHANNEL_ID 
        
        f = self.attributes[V_ID][height_id, width_id, channel_id, 0:self.attributes[NUMPRED_ID] + 1]
        xmin = f[0]
        xmax = f[0]
        
        for i in range(1, self.attributes[NUMPRED_ID] + 1):
            if f[i] >= 0:
                xmin = xmin + f[i] * self.attributes[PREDLB_ID][i - 1]
                xmax = xmax + f[i] * self.attributes[PREDUB_ID][i - 1]
            else:
                xmin = xmin + f[i] * self.attributes[PREDUB_ID][i - 1]
                xmax = xmax + f[i] * self.attributes[PREDLB_ID][i - 1]

        return np.array([xmin, xmax])

    def estimate_ranges(self, dis_opt = DEFAULT_DISP_OPTION):
        """
            Estimates the ranges using only a predicate bound information
            
            dis_opt : string -> display option
            
            return -> [image_lb, image_ub]
        """
        
        assert (not self.isempty(self.attributes[C_ID]) and not self.isempty(self.attributes[D_ID])), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        if self.isempty(self.attributes[IM_LB_ID]) or self.isempty(self.attributes[IM_UB_ID]):
            image_lb = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))
            image_ub = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))
            
            size = self.attributes[HEIGHT_ID] * self.attributes[WIDTH_ID] * self.attributes[NUM_CHANNEL_ID]
            
            disp_flag = False
            if dis_opt == DISPLAY_ON_OPTION:
                disp_flag = True
                print(ESTIMATE_RANGE_STAGE_STARTED)
                
            for i in range(self.attributes[HEIGHT_ID]):
                for j in range(self.attributes[WIDTH_ID]):
                    for k in range(self.attributes[NUM_CHANNEL_ID]):
                        image_lb[i, j, k], image_ub[i, j, k] = self.estimate_range(i+1, j+1, k+1)
                        
                        if disp_flag:
                            print(ESTIMATE_RANGE_STAGE_OVER)
                
            self.attributes[IM_LB_ID] = image_lb
            self.attributes[IM_UB_ID] = image_ub
        else:
            image_lb = self.attributes[IM_LB_ID]
            image_ub = self.attributes[IM_UB_ID]
            
        return np.array([image_lb, image_ub])
    
    def get_ranges(self, dis_opt = DEFAULT_DISP_OPTION):
        """
            Computes the lower and upper bound images of the ImageStar
            
            return -> [image_lb : np.array([]) -> lower bound image,
                       image_ub : np.array([]) -> upper bound image]
        """
                
        image_lb = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))
        image_ub = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))

        size = self.attributes[HEIGHT_ID] * self.attributes[WIDTH_ID] * self.attributes[NUM_CHANNEL_ID]
            
        disp_flag = False
        if dis_opt == DISPLAY_ON_OPTION:
            disp_flag = True
            print(ESTIMATE_RANGE_STAGE_STARTED)
                
        for i in range(self.attributes[HEIGHT_ID]):
            for j in range(self.attributes[WIDTH_ID]):
                for k in range(self.attributes[NUM_CHANNEL_ID]):
                    image_lb[i, j, k], image_ub[i, j, k] = self.get_range(i + 1, j + 1, k + 1)
                        
                    if disp_flag:
                        print(ESTIMATE_RANGE_STAGE_OVER)
                
        self.attributes[IM_LB_ID] = image_lb
        self.attributes[IM_UB_ID] = image_ub
            
        return np.array([image_lb, image_ub])
            
    def update_ranges(self, *args):
        """
            Updates local ranges for the MaxPooling operation
            
            points : np.array([*]) -> local points = [x1 y1 c1; x2 y2 c2; ...]
        """
        updated_ranges = []
        
        for i in range(len(args[POINTS_ID])):
            updated_ranges.append(self.get_range(args[POINTS_ID][i][0], args[POINTS_ID][i][1], args[POINTS_ID][i][2]))
                                  
        return updated_ranges
            
        
            
    def get_num_attacked_pixels(self):
        """
            Computes the number of attacked pixels in the ImageStar
            
            return : int -> the number of pixels
        """
        
        V1 = np.zeros((self.attributes[HEIGHT_ID], self.attributes[WIDTH_ID], self.attributes[NUM_CHANNEL_ID]))
        V3 = V1
        
        for i in range(1, self.attributes[NUMPRED_ID] + 1):
            V2 = (self.attributes[V_ID][:,:,:,i] != V1)
            V3 = V3 + V2
            
        V4 = np.amax(V3, 2)
        return sum(sum(V4))
    
    def get_local_bound(self, *args):
        """
            Computes the local bound for the Max Pooling operation
            
            args : np.array([]) that includes =>
            start_point : np.array([x1, y1]) -> the start point of the local (partial) image
            pool_size : np.array([height, width]) -> the height and width of max pooling
            channel_id : int -> the index of the channel
            
            return : [lb : np.array([*]) -> the lower bound of all points in the local region,
                      ub : np.array([*]) -> the upper bound of all points in the local region]
            
        """
        
        points = self.get_local_points(args[START_POINT_ID], args[POOL_SIZE_ID])
        points_num = len(points)
        
        if self.isempty(self.attributes[IM_LB_ID]) or self.isempty(self.attributes[IM_UB_ID]):
            image_lb, image_ub = self.get_ranges()
        else:
            image_lb = self.attributes[IM_LB_ID]
            image_ub = self.attributes[IM_UB_ID]
        
        lb = image_lb[int(points[0,0]), int(points[0,1]), self.attributes[NUM_CHANNEL_ID] - 1]
        ub = image_ub[int(points[0,0]), int(points[0,1]), self.attributes[NUM_CHANNEL_ID] - 1]
        
        for i in range(1, points_num):
            if image_lb[int(points[i,0]), int(points[i,1]), self.attributes[NUM_CHANNEL_ID] - 1] < lb:
                lb = image_lb[int(points[i,0]), int(points[i,1]), self.attributes[NUM_CHANNEL_ID] - 1]
            
            if image_ub[int(points[i,0]), int(points[i,1]), self.attributes[NUM_CHANNEL_ID] - 1] > ub:
                ub = image_ub[int(points[i,0]), int(points[i,1]), self.attributes[NUM_CHANNEL_ID] - 1]
                
        return [lb, ub]
            
    def get_local_points(self, start_point, pool_size):
        """
            Computes all local points indices for Max Pooling operation
            
            start_point : np.array([x1, y1]) -> the start point of the local (partial) image
            pool_size : np.array([height, width]) -> the height and width of max pooling
            
            returns : np.array([*]) -> all indices of the points for a single max pooling operation
                                       (includeing the start point)
        """
            
        x0 = start_point[0] # vertical index of the startpoint
        y0 = start_point[1] # horizontal index of the startpoint
        
        h = pool_size[0] # height of the MaxPooling layer
        w = pool_size[1] # width of the MaxPooling layer
        
        assert (x0 >= 0 and y0 >= 0 and x0 + h - 1 < self.attributes[HEIGHT_ID] \
                        and y0 + w - 1 < self.attributes[WIDTH_ID]), \
                        'error: %s' % ERRMSG_INVALID_STARTPOINT_POOLSIZE 
                        
        points = np.zeros((h * w, 2))
        
        for i in range(h):
            if i == 0:
                x1 = x0
            else:
                x1 = x1 + 1
                
            for j in range(w):
                if j==0:
                    y1 = y0;
                else:
                    y1 = y1 + 1;
                    
                points[i * w + j, :] = np.array([x1, y1])
                
        return points - 1
      
    def get_localMax_index(self, *args):
        """
            Gets local max index. Attempts to find the maximum point of the local image.
            It's used in over-approximate reachability analysis of the maxpooling operation
        
            startpoints : np.array([int, int]) -> startpoint of the local image
            pool_size : np.array([int, int]) -> the height and width of the max pooling layer
            channel_id : int -> the channel index
            
            return -> max_id
        """
        
        points = self.get_local_points(args[START_POINT_ID], args[POOL_SIZE_ID])
    
        if self.isempty(self.attributes[IM_LB_ID]) or self.isempty(self.attributes[IM_UB_ID]):
            self.estimate_ranges()
    
        height = args[POOL_SIZE_ID][0]
        width = args[POOL_SIZE_ID][0]
        size = height * width
        
        lb = np.zeros((size, 1))
        ub = np.zeros((size, 1))
        
        for i in range(size):
            current_point = points[i, :].astype(int)
            
            lb[i] = self.attributes[IM_LB_ID][current_point[0], current_point[1], args[CHANNEL_ID] - 1]
            ub[i] = self.attributes[IM_UB_ID][current_point[0], current_point[1], args[CHANNEL_ID] - 1]
        
            
        [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
        
        a = np.argwhere((ub - max_lb_val) > 0)[:,0]
        a1 = np.argwhere((ub - max_lb_val) >= 0)[:,0]
        a = np.delete(a, np.argwhere(a==max_lb_id)[:,0])
        
        if self.isempty(a):
            max_id = points[max_lb_id, :]
        else:
            candidates = a1
            
            candidates_num = len(candidates)
            
            new_points = []
            new_points1 = np.zeros((candidates_num, 2))
            
            for i in range(candidates_num):
                selected_points = points[candidates[i], :]
                new_points.append(np.append(selected_points, args[CHANNEL_ID] - 1))
                new_points1[i, :] = selected_points
               
            self.update_ranges(new_points)
            
            lb = np.zeros((candidates_num,1))
            ub = np.zeros((candidates_num,1))
            
            for i in range(candidates_num):
                #TODO: THIS SHOULD BE INITIALLY INT
                current_point = points[candidates[i], :]
                
                lb[i] = self.attributes[IM_LB_ID][int(current_point[0]), int(current_point[1]), int(args[CHANNEL_ID])]
                ub[i] = self.attributes[IM_UB_ID][int(current_point[0]), int(current_point[1]), int(args[CHANNEL_ID])]
                
            [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
            
            a = np.argwhere((ub - max_lb_val) > 0)[:,0]
            a = np.delete(a, np.argwhere(a==max_lb_id)[:,0])
            
            if self.isempty(a):
                max_id = new_points1[max_lb_val, :]
            else:
                candidates1 = (ub - max_lb_val) >= 0
                max_id = new_points1[max_lb_id, :]
                
                candidates1[candidates1 == max_lb_id] == []
                candidates_num = len(candidates1)
                
                max_id1 = max_id
                
                for j in range(candidates_num):
                    p1 = new_points[candidates1[j], :]
                    
                    if self.is_p1_larger_p2(np.array([p1[0], p2[1], args[CHANNEL_ID]]), \
                                            np.array([max_id[0], max_id[1], args[CHANNEL_ID]])):
                        max_id1 = np.array([max_ids1, p1])
                        
                        
                max_id = max_id1
                        
                print('\nThe local image has %d max candidates: \t%d' % np.size(max_id, 0))
                
        return np.append(max_id, args[CHANNEL_ID] * np.zeros((len(max_id.shape)))).tolist()
          
    def get_localMax_index2(self, start_point, pool_size, channel_id):
        """
            Gets local max index. Attempts to find the maximum point of the local image.
            It's used in over-approximate reachability analysis of the maxpooling operation
        
            startpoints : np.array([int, int]) -> startpoint of the local image
            pool_size : np.array([int, int]) -> the height and width of the max pooling layer
            channel_id : int -> the channel index
            
            return -> max_id
        """
        
        points = self.get_local_points(start_point, pool_size)
        
        height = pool_size[0]
        width = pool_size[1]
        size = height * width
        
        lb = np.zeros((size,1))
        ub = np.zeros((size,1))
        
        for i in range(size):
            current_point = points[i, :].astype(int)
            
            lb[i] = self.attributes[IM_LB_ID][current_point[0], current_point[1], channel_id]
            ub[i] = self.attributes[IM_UB_ID][current_point[0], current_point[1], channel_id]
        
            
        [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
            
        max_id = np.argwhere((ub - max_lb_val) > 0)[:,0]
        
        return np.append(max_id, channel_id * np.zeros((len(max_id.shape)))).tolist()

    def add_max_id(self, name, max_id):
        """
            Adds a matrix used for unmaxpooling reachability
            
            name : string -> name of the max pooling layer
            max_id : np.array([*int]) -> max indices
        """
            
        #TODO: WHAT IS 'A' in NNV?
        return Exception("unimplemented")
    
    def update_max_id(self, name, max_id, pos):
        """
            Updates a matrix used for unmaxpooling reachability
        
            name : string -> name of the max pooling layer
            max_id : np.array([*int]) -> max indices
            pos : np.array([]) -> the position of the local pixel of the max map
                                  corresponding to the max_id
        """
        
        ids_num = len(self.attributes[MAX_IDS])
        
        unk_layer_num = 0
        
        for i in range(ids_num):
            if self.attributes[MAX_IDS][i].get_name() == name:
                self.attributes[MAX_IDS][i].get_max_ids()[pos[0], pos[1], pos[2]] = max_id
                break
            else:
                unk_layer_num += 1
                
        if unk_laye_num == ids_num:
            raise Exception('error: %s' % ERRMSG_UNK_NAME_MAX_POOL_LAYER)

    def add_input_size(self, name, input_size):
        """
            Adds a matrix used for unmaxpooling reachability
            
            name : string -> name of the max pooling layer
            input_size : np.array([*int]) -> input size of the original image
        """
         
        #TODO: WHAT IS 'A' in NNV?    
        return Exception("unimplemented")
        
    def is_p1_larger_p2(self, *args):
        """
            Compares two specific points in the image. Checks if p1 > p2 is feasible.
            Can be used in max pooling operation
            
            p1 : np.array([*int]) - the first points = [h1, w1, c1]
            p2 : np.array([*int]) - the second points = [h2, w2, c2], where
                                    h - height, w - width, c - channel id
                                    
            return : bool -> 1 if p1 > p2 is feasible,
                             0 if p1 > p2 is not feasible
            
        """
        
        C1 = np.zeros(self.attributes[NUMPRED_ID])
        
        for i in range(1, self.attributes[NUMPRED_ID] + 1):
            C1[i-1] = self.attributes[V_ID][args[P2_ID][0], args[P2_ID][1], args[P2_ID][2], i] - \
                      self.attributes[V_ID][args[P1_ID][0], args[P1_ID][1], args[P1_ID][2], i]
                      
        # TODO: WHY DOESN'T THE SUBTRAHEND HAVE THE 4-TH COMPONENT INDEXING IN NNV?
        d1 = self.attributes[V_ID][args[P1_ID][0], args[P1_ID][1], args[P1_ID][2], 0] - \
                 self.attributes[V_ID][args[P2_ID][0], args[P2_ID][1], args[P2_ID][2], 0]
                 
        new_C = np.vstack((self.attributes[C_ID], C1))
        new_d = np.vstack((self.attributes[D_ID], d1))
        
        S = Star(self.attributes[V_ID], new_C, new_d, self.attributes[PREDLB_ID], self.attributes[PREDUB_ID])

        if S.isEmptySet():
            return 0
        else:
            return 1
     
    def is_max(self, *args):
        """
            Checks if a pixel value is the maximum value compared with the other ones.
            Implements one of the core steps of the maxpooling operation over
            the ImageStar
            
            max_map -> the current max_map of the ImageStar
            ori_image -> the original ImageStar to compute the max_map
            center -> the center pixel position that is checked
                      center = [x1, y1, c1]
            others -> the positions of the pixels we want to compare the center against
                      others = [x2, y2, c3; x3, y3, c3]
            out_image : ImageStar -> the updated input image
            
            return -> a new predicate
        """
        
        size = args[OTHERS_ID].shape[0]
        
        new_C = np.zeros((size, args[MAX_MAP_ID].get_num_pred()))
        new_d = np.zeros((size, 1))
        
        for i in range(n):
            new_d[i] = args[ORI_IMAGE_ID].get_V()[args[CENTER_ID][0], args[CENTER_ID][1], args[CENTER_ID][2], 0] - \
                       args[ORI_IMAGE_ID].get_V()[args[OTHERS_ID][0], args[OTHERS_ID][1], args[OTHERS_ID][2], 0]
        
            for j in range(args[MAX_MAP_ID].get_num_pred()):
                new_C[i, j] = args[ORI_IMAGE_ID].get_V()[args[CENTER_ID][0], args[CENTER_ID][1], args[CENTER_ID][2], j + 1] - \
                       args[ORI_IMAGE_ID].get_V()[args[OTHERS_ID][0], args[OTHERS_ID][1], args[OTHERS_ID][2], j + 1]
        
        C1 = np.vstack(args[MAX_MAP_ID].get_C(), new_C)
        d1 = np.vstack(args[MAX_MAP_ID].get_D(), new_d)
        
        # TODO: remove redundant constraints here
        
        return C1, d1
    
    def reshape(self, input, new_shape):
        """
            Reshapes the ImageStar
            
            input : ImageStar -> the input ImageStar
            new_shape : np.array([]) -> new shape
            
            return -> a reshaped ImageStar
        """
        
        size = np.size(new_shape)
        
        assert size[1] == 3 and size[0] == 1, 'error: %s' % ERRMSG_INVALID_NEW_SHAPE
        
        assert np.multiply(new_shape[:]) == input.get_height() * input.get_width() * input.get_num_channel(), \
               'error: %s' % ERRMSG_SHAPES_INCONSISTENCY
               
        new_V = np.reshape(input.get_V(), (new_shape, input.get_num_pred() + 1))
        
        return ImageStar(new_V, input.get_C(), input.get_d(), \
                         input.get_pred_lb(), input.get_pred_ub,
                         input.get_im_lb, input.get_im_ub)
        
    def add_constraints(self, input, p1, p2):
        """
            Adds a new constraint to predicate variables of an ImageStar
            used for finding counter examples. Add a new constraint: p2 >= p1
            
            input : ImageStar -> an input ImageStar
            p1 : np.array([]) -> first point position
            p2 : np.array([]) -> second point position
            new_C : np.array([]) -> a new predicate constraint matrix C
            new_d : new predicate constraint vector
            
            return -> [new_C, new_d] - a new predicate
        """
        
        assert input is ImageStar, 'error: %s' % ERRMSG_INPUT_NOT_IMAGESTAR
        
        new_d = input.get_V()[p2[0], p2[1], p2[2], 1] - input.get_V()[p2[0], p2[1], p2[2], 1]
        new_C = input.get_V()[p2[0], p2[1], p2[2], 1:input.get_num_pred() + 1]
        
        new_C = np.reshape(new_c, (1, input.get_num_pred()))
        
        new_C = np.vstack(input.get_C(), new_C)
        new_C = np.vstack(input.get_d(), new_d)
        
        return new_C, new_d
        
##################### GET/SET METHODS #####################
        
    def get_V(self):
        """
            return -> the center and the basis matrix of the ImageStar
        """
        
        return self.attributes[V_ID]
        
########################## UTILS ##########################

    def validate_params(self, params):
        for param in params:
            assert isinstance(param, np.ndarray), 'error: ImageStar does not support parameters of dtype = %s' % param.dtype

    def isempty_init(self, *params):
        flag = True
        
        for param in params:
            flag = self.isempty(np.array(param))
            
            if flag == False:
                break
            
        return flag
            
    def isempty(self, param):
        return param.size == 0 or (param is np.array and param.shape[0] == 0)
            
    def init_empty_imagestar(self):
        for i in range(len(self.attributes)):
            if self.is_scalar_attribute(i):
                self.attributes[i] = 0
            else:
                self.attributes[i] = np.array([])

    def copy_deep(self, imagestar):
        for i in range(len(self.attributes)):
            self.attributes[i] = imagestar.get_attribute(i)

    def validate_point_dim(self, point, height, width):
        return (point[0] > -1) and (point[0] <= self.attributes[HEIGHT_ID]) and \
               (point[1] > -1) and (point[1] <= self.attributes[WIDTH_ID]) and \
               (point[2] > -1) and (point[2] <= self.attributes[NUM_CHANNEL_ID])

    def offset_args(self, args, offset):
        result = []
        
        for i in range(len(args) + offset):
            result.append(np.array([]))
            
            if i >= offset:
                result[i] = args[i - offset]
                
        return result
           
    def is_scalar(self, param):
        return isinstance(param, np.ndarray)
    
    def is_scalar_attribute(self, attribute_id):
        return attribute_id in self.scalar_attributes_ids
