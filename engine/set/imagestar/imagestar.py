#!/usr/bin/python3
import numpy as np
import gurobipy as gp
from gurobipy import GRB 
import sys, os
import copy

import sys
import operator

sys.path.insert(0, "../star")
from star import *

sys.path.insert(0, "../zono")
from zono import *

############################ ERROR MESSAGES ############################

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

ERRMSG_SHAPES_INCONSISTENCY = "New shape is inconsistent with the current shape"

ERRMSG_INPUT_NOT_IMAGESTAR = "Input set is not an ImageStar"

ERRMSG_UNK_MP_LAYER_NAME = "Unknown name of the maxpooling layer"

ESTIMATE_RANGE_STAGE_STARTED = "Ranges estimation started..."
ESTIMATE_RANGE_STAGE_OVER = "Ranges estimation finished..."

ERRMSG_INCONSISTENT_SOLVER_INPUT = "The given solver is not supported. Use \'glpk\' for GNU Linear Programming Kit or \'linprog\' for Gurobi"

DEFAULT_DISP_OPTION = []

class ImageStar:
    """Class for representing set of images using Star set
     An image can be attacked by bounded noise. An attacked image can
     be represented using an ImageStar Set
     author: Mykhailo Ivashchenko
     date: 2/14/2022
    
    =================================================================\n
       A 3-channels color image is represented by 3-dimensional array 
       Each dimension contains a h x w matrix, h and w is the height
       width of the image. h * w = number of pixels in the image.
       *A gray image has only one channel.
    
       Problem: How to represent a disturbed(attacked) image?
       
       Use a center image (a matrix) + a disturbance matrix (positions
       of attacks and bounds of corresponding noises)
    
       For example: Consider a 4 x 4 (16 pixels) gray image 
       The image is represented by 4 x 4 matrix:
                   IM = [1 1 0 1; 0 1 0 0; 1 0 1 0; 0 1 1 1]
       This image is attacked at pixel (1,1) (1,2) and (2,4) by bounded
       noises:     |n1| <= 0.1, |n2| <= 0.2, |n3| <= 0.05
    
    
       Lower and upper noises bounds matrices are: 
             LB = [-0.1 -0.2 0 0; 0 0 0 -0.05; 0 0 0 0; 0 0 0 0]
             UB = [0.1 0.2 0 0; 0 0 0 0.05; 0 0 0 0; 0 0 0 0]
       The lower and upper bounds matrices also describe the position of 
       attack.
    
       Under attack we have: -0.1 + 1 <= IM(1,1) <= 1 + 0.1
                             -0.2 + 1 <= IM(1,2) <= 1 + 0.2
                                -0.05 <= IM(2,4) <= 0.05
    
       To represent the attacked image we use IM, LB, UB matrices
       For multi-channel image we use multi-dimensional array IM, LB, UB
       to represent the attacked image. 
       For example, for an attacked color image with 3 channels we have
       IM(:, :, 1) = IM1, IM(:,:,2) = IM2, IM(:,:,3) = IM3
       LB(:, :, 1) = LB1, LB(:,:,2) = LB2, LB(:,:,3) = LB3
       UB(:, :, 1) = UB1, UB(:,:,2) = UB2, UB(:,:,3) = UB3
       
       The image object is: image = ImageStar(IM, LB, UB)
    =================================================================\n
    
    **2D representation of an ImageStar**
    =================================================================\n
                       Definition of Star2D
     
     A 2D star set S is defined by: 
     S = {x| x = V[0] + a[1]*V[1] + a[2]*V[2] + ... + a[n]*V[n]
               = V * b, V = {c V[1] V[2] ... V[n]}, 
                        b = [1 a[1] a[2] ... a[n]]^T                                   
                        where C*a <= d, constraints on a[i]}
     where, V[0], V[i] are 2D matrices with the same dimension, i.e., 
     V[i] \in R^{m x n}
     V[0] : is called the center matrix and V[i] is called the basic matrix 
     [a[1]...a[n] are called predicate variables
     C: is the predicate constraint matrix
     d: is the predicate constraint vector
     
     The notion of Star2D is more general than the original Star set where
     the V[0] and V[i] are vectors. 
     
     Dimension of Star2D is the dimension of the center matrix V[0]
     
    =================================================================\n"""


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
        
        self.init_attributes()
                        
        if len(args) == 7 or len(args) == 5:
            v = args[0]
            C = args[1]
            d = args[2]
            pred_lb = args[3]
            pred_ub = args[4]
            
            if np.size(v) and np.size(C) and np.size(d) and np.size(pred_lb) and np.size(pred_ub):
                assert (C.shape[0] == 1 and np.size(d) == 1) or (C.shape[0] == d.shape[0]), \
                       'error: %s' % ERRMSG_INCONSISTENT_CONSTR_DIM
                
                assert (np.size(d) == 1) or (len(np.shape(d)) == 1) or (len(np.shape(d)) == 2 and np.shape(d)[1] == 1), 'error: %s' % ERRMSG_INVALID_CONSTR_VEC
                
                self.numpred = C.shape[1]
                
                assert C.shape[1] == pred_lb.shape[0] == pred_ub.shape[0], 'error: %s' % ERRMSG_INCONSISTENT_PRED_BOUND_DIM
                
                self.C = C.astype('float64')
                self.d = d.astype('float64')
                
                
                assert len(pred_lb.shape) == len(pred_ub.shape) == 1 or pred_ub.shape[1], 'error: %s' % ERRMSG_INCONSISTENT_BOUND_DIM
                
                self.pred_lb = pred_lb.astype('float64')
                self.pred_ub = pred_ub.astype('float64')
                
                n = v.shape
                
                if len(n) < 2:
                    raise Exception('error: %s' % ERRMSG_INVALID_BASE_MATRIX)
                else:
                    self.height = n[0]
                    self.width = n[1]
                    
                    self.V = v
                    
                    if len(n) == 4:
                        assert n[3] == self.numpred + 1, 'error: %s' % ERRMSG_INCONSISTENT_BASIS_MATRIX_PRED_NUM
                        
                        self.num_channels = n[2]
                    else:
                        # TODO: ASK WHY THIS HAPPENS AFTER THE ASSIGNMENT IN LINE 205
                        #self.numpred = 0
                        
                        if len(n) == 3:
                            self.num_channels = n[2]
                        elif len(n) == 2:
                            self.num_channels = 1
                
                if len(args) == 7:
                    im_lb = args[6]
                    im_ub = args[7]
                    
                    if im_lb.shape[0] != 0 and (im_lb.shape[0] != self.height or im_lb.shape[1] != self.width):
                        raise Exception('error: %s' % ERRMSG_INCONSISTENT_LB_DIM)
                    else:
                        self.im_lb = im_lb.astype('float64')      
                        
                    if im_ub.shape[0] != 0 and (im_ub.shape[0] != self.height or im_ub.shape[1] != self.width):
                        raise Exception('error: %s' % ERRMSG_INCONSISTENT_UB_DIM)
                    else:
                        self.im_ub = im_ub.astype('float64')
                    
        elif len(args) == 3:
            im = args[0]
            lb = args[1]
            ub = args[2]
            
            if np.size(im) and np.size(lb) and np.size(ub) and self.V.shape[0] == 0:
                n = im.shape
                l = lb.shape
                u = ub.shape
                
                assert (n[0] == l[0] == u[0] and n[1] == l[1] == u[1]) and (len(n) == len(l) == len(u)), 'error: %s' % ERRMSG_INCONSISTENT_CENTER_IMG_ATTACK_MATRIX
                            
                assert len(n) == len(l) == len(u), 'error: %s' % ERRMSG_INCONSISTENT_CHANNELS_NUM
                
                self.im = im.astype('float64')
                self.lb = lb.astype('float64')
                self.ub = ub.astype('float64')
                
                self.height = n[0]
                self.width = n[1]
                
                if len(n) == 2:
                    self.num_channels = 2
                elif len(n) == 3:
                    self.num_channels = 3
                else:
                    raise Exception('error: %s' % ERRMSG_INCONSISTENT_CHANNELS_NUM)
                
                self.im_lb = self.im + self.lb
                self.im_ub = self.im + self.ub
                
                n = self.im_lb.shape
                
                I = 0
                
                if len(n) == 3:
                    I = Star(self.im_lb.flatten(order=self.flatten_mode), self.im_ub.flatten(order=self.flatten_mode))
                    self.V = np.reshape(I.V, (I.nVar + 1, n[0] * n[1] * n[2]))
                else:
                    I = Star(self.im_lb.flatten(order=self.flatten_mode), self.im_ub.flatten(order=self.flatten_mode))
                    self.V = np.reshape(I.V, (I.nVar + 1, n[0] * n[1]))
                    
                self.C = I.C
                self.d = I.d
                
                # TODO: ask why does Star have predicate_lb and ImageStar pred_lb?
                self.pred_lb  = I.predicate_lb
                self.pred_ub = I.predicate_ub
                
                self.numpred = I.nVar
        elif len(args) == 2:                
            im_lb = args[0]
            im_ub = args[1]
            
            if np.size(im_lb) and np.size(im_ub) and self.V.shape[0] == 0:
                lb_shape = im_lb.shape
                ub_shape = im_ub.shape
                
                assert len(lb_shape) == len(ub_shape), 'error: %s' % ERRMSG_INCONSISTENT_LB_UB_DIM
                
                for i in range(len(lb_shape)):
                    assert lb_shape[i] == ub_shape[i], 'error: %s' % ERRMSG_INCONSISTENT_LB_UB_DIM
                    
                lb = im_lb.flatten(order=self.flatten_mode)
                ub = im_ub.flatten(order=self.flatten_mode)
                
                S = Star(lb, ub)
                    
                self.copy_deep(S.toImageStar(lb_shape[0], lb_shape[1], (lb_shape[2] if len(lb_shape) == 3 else 1)))
                    
                self.im_lb = im_lb.astype('float64')
                self.im_ub = im_ub.astype('float64')
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
        
        assert (not self.isempty(self.V)), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        if self.isempty(self.C) or self.isempty(self.d):
            return self.im
        else:
            new_V = np.hstack((np.zeros((self.numpred, 1)), np.eye(self.numpred)))
            S = Star(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
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
        
        assert (not self.isempty(self.V)), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        assert len(pred_val.shape) == 1, 'error: %s' % ERRMSG_INVALID_PREDICATE_VEC
        
        assert pred_val.shape[0] == self.numpred, 'error: %s' % ERRMSG_INCONSISTENT_PREDVEC_PREDNUM
        
        image = np.zeros((self.height, self.width, self.num_channels))
        
        for i in range(self.num_channels):
            image[:, :, i] = self.V[:, :, i, 1]
            
            for j in range(2, self.numpred + 1):
                image[:, :, i] = image[:, :, i] + pred_val[j - 1] * self.V[:, :, i, j]
 
        return image
 
    def affine_map(self, scale, offset):
        """
            Performs affine mapping of the ImageStar: y = scale * x + offset
            
            scale : *float -> scale coefficient [1 x 1 x num_channel] array
            offset : *float -> offset coefficient [1 x 1 x num_channel] array
            return -> a new ImageStar
        """
 
        assert (self.isempty(scale) or self.is_scalar(scale) or len(scale.shape) == self.num_channels), 'error: %s' % ERRMSG_INCONSISTENT_SCALE_CHANNELS_NUM
        
        new_V = 0
        
        if not self.isempty(scale):
            new_V = np.multiply(scale, self.V)
        else:
            new_V = self.V
        
        # Affine Mapping changes the center
        if not self.isempty(offset):
            new_V[:, :, :, 0] = new_V[:, :, :, 0] + offset
            
        return ImageStar(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
    
    def to_star(self):
        from star import Star
        """
            Converts current ImageStar to Star
            
            return -> created Star
        """
 
        pixel_num = self.height * self.width * self.num_channels
        
        new_V = np.zeros((pixel_num, self.numpred + 1))
        
        if self.isempty(new_V):
            # TODO: error: failed to create Star set
            return Star()
        else:
                    
            v_shape = self.V.shape
                    
            if(len(v_shape) == 3):
                 self.V = np.reshape(self.V, (v_shape[0], v_shape[1], 1, v_shape[2]))
                 
            for i in range(self.num_channels):
                for j in range(self.numpred + 1):        
                    new_V[:, j] = self.V[:, :, i, j].flatten(order=self.flatten_mode)
                
            if not self.isempty(self.im_lb) and not self.isempty(self.im_ub):
                state_lb = self.im_lb.flatten(order=self.flatten_mode)
                state_ub = self.im_ub.flatten(order=self.flatten_mode)
                
                # TODO: error: failed to create Star set
                S = Star(new_V, self.C, self.d, self.pred_lb, self.pred_ub, state_lb, state_ub)
            else:
                # TODO: error: failed to create Star set
                S = Star(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
                
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
            assert (img_size[0] == self.height and img_size[1] == self.width and self.num_channels == 1), 'error: %s' % ERRMSG_INCONSISTENT_IMGDIM_IMGSTAR
        elif len(img_size) == 3:
            assert (img_size[0] == self.height and img_size[1] == self.width and img_size[2] == self.num_channels), 'error: %s' % ERRMSG_INCONSISTENT_IMGDIM_IMGSTAR
        else:
            raise Exception('error: %s' % ERRMSG_INVALID_INPUT_IMG)
            
        image_vec = image.flatten(order=self.flatten_mode)
        
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
        assert self.validate_point_dim(point1, self.height, self.width), 'error: %s' % ERRMSG_INVALID_FIRST_INPUT_POINT
        assert self.validate_point_dim(point2, self.height, self.width), 'error: %s' % ERRMSG_INVALID_SECOND_INPUT_POINT
        
        n = self.numpred + 1
        
        new_V = np.zeros((2, n))
        
        for i in range(n):
            new_V[0, i] = self.V[point1[0], point1[1], point1[2], i]
            new_V[1, i] = self.V[point2[0], point2[1], point2[2], i]
            
        return Star(new_V, self.C, self.d, self.pred_lb, self.pred_ub)
        
        
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
        
        vert_id = None
        horiz_id = None
        channel_id = None
        solver = None
        options = None
        
        if len(args) == 3:
            vert_id = args[0]
            horiz_id = args[1]
            channel_id = args[2]
            solver = 'glpk'
            options = []
        elif len(args) == 4:
            vert_id = args[0]
            horiz_id = args[1]
            channel_id = args[2]
            solver = args[3]
            options = []
        elif len(args) == 5:
            vert_id = args[0]
            horiz_id = args[1]
            channel_id = args[2]
            solver = args[3]
            options = args[4]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 3, 4 or 5')
            
        assert (not self.isempty(self.C) and not self.isempty(self.d)), 'error: %s' % ERRMSG_IMGSTAR_EMPTY        
        assert (vert_id > -1 and vert_id < self.height), 'error: %s' % ERRMSG_INVALID_VERT_ID
        assert (horiz_id > -1 and horiz_id < self.width), 'error: %s' % ERRMSG_INVALID_HORIZ_ID
        assert (channel_id > -1 and channel_id < self.num_channels), 'error: %s' % ERRMSG_INCONSISTENT_CHANNELS_NUM
        self.validate_solver_name(solver)
        self.validate_options_list(options)
        
        bounds = [np.array([]), np.array([])]
        
        f = self.V[vert_id, horiz_id, channel_id, 1:self.numpred + 1]
        
        if (f == 0).all():
            bounds[0] = self.V[vert_id, horiz_id, channel_id, 0]
            bounds[1] = self.V[vert_id, horiz_id, channel_id, 0]
        elif solver == 'linprog':
            if 'disp' in options:
                print('Calculating the lower and upper bounds using Gurobi...')
                
            min_ = gp.Model()
            min_.Params.LogToConsole = 0
            min_.Params.OptimalityTol = 1e-9
            if self.pred_lb.size and self.pred_ub.size:
                x = min_.addMVar(shape=self.numpred, lb=self.pred_lb, ub=self.pred_ub)
            else:
                x = min_.addMVar(shape=self.numpred)
            min_.setObjective(f @ x, GRB.MINIMIZE)
            C = sp.csr_matrix(self.C)
            d = np.array(self.d).flatten(order=self.flatten_mode)
            min_.addConstr(C @ x <= d)
            min_.optimize()

            if min_.status == 2:
                xmin = min_.objVal + self.V[vert_id, horiz_id, channel_id, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (min_.status))

            max_ = gp.Model()
            max_.Params.LogToConsole = 0
            max_.Params.OptimalityTol = 1e-9
            if self.pred_lb.size and self.pred_ub.size:
                x = max_.addMVar(shape=self.numpred, lb=self.pred_lb, ub=self.pred_ub)
            else:
                x = max_.addMVar(shape=self.numpred)
            max_.setObjective(f @ x, GRB.MAXIMIZE)
            C = sp.csr_matrix(self.C)
            d = np.array(self.d).flatten(order=self.flatten_mode)
            max_.addConstr(C @ x <= d)
            max_.optimize()

            if max_.status == 2:
                xmax = max_.objVal + self.V[vert_id, horiz_id, channel_id, 0]
            else:
                raise Exception('error: cannot find an optimal solution, exitflag = %d' % (max_.status))
        else:
            # TODO: add glpk here
            if 'disp' in options:
                print('Calculating the lower and upper bounds using GLPK...')
                
            print('GLPK is not implemented for this method yet')

        return np.array([xmin, xmax])

    def estimate_range(self, height_id, width_id, channel_id):
        """
            Estimates a range using only a predicate bounds information
            
            h : int -> height index
            w : int -> width index
            c : int -> channel index
            
            return -> [xmin, xmax]
        """

        assert (not self.isempty(self.C) and not self.isempty(self.d)), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        
        height_id = int(height_id)
        width_id = int(width_id)
        channel_id = int(channel_id)
        
        assert (height_id > -1 and height_id < self.height), 'error: %s' % ERRMSG_INVALID_VERT_ID
        assert (width_id > -1 and width_id < self.width), 'error: %s' % ERRMSG_INVALID_HORIZ_ID
        assert (channel_id > -1 and channel_id < self.num_channels), 'error: %s' % ERRMSG_INVALID_CHANNEL_ID 
        
        f = self.V[height_id, width_id, channel_id, 0:self.numpred + 1]
        xmin = f[0]
        xmax = f[0]
        
        for i in range(1, self.numpred + 1):
            if f[i] >= 0:
                xmin = xmin + f[i] * self.pred_lb[i - 1]
                xmax = xmax + f[i] * self.pred_ub[i - 1]
            else:
                xmin = xmin + f[i] * self.pred_ub[i - 1]
                xmax = xmax + f[i] * self.pred_lb[i - 1]

        return np.array([xmin, xmax])

    def estimate_ranges(self, options = DEFAULT_DISP_OPTION):
        """
            Estimates the ranges using only a predicate bound information
            
            options : string -> a set of options
            
            return -> [image_lb, image_ub]
        """
                
        assert (not self.isempty(self.C) and not self.isempty(self.d)), 'error: %s' % ERRMSG_IMGSTAR_EMPTY
        self.validate_options_list(options)
        
        if self.isempty(self.im_lb) or self.isempty(self.im_ub):
            image_lb = np.zeros((self.height, self.width, self.num_channels))
            image_ub = np.zeros((self.height, self.width, self.num_channels))
            
            size = self.height * self.width * self.num_channels
            
            if 'disp' in options:
                print(ESTIMATE_RANGE_STAGE_STARTED)
                
            for i in range(self.height):
                for j in range(self.width):
                    for k in range(self.num_channels):
                        image_lb[i, j, k], image_ub[i, j, k] = self.estimate_range(i, j, k)
                        
                        if 'disp' in options:
                            print(ESTIMATE_RANGE_STAGE_OVER)
                
            self.im_lb = image_lb
            self.im_ub = image_ub
        else:
            image_lb = self.im_lb
            image_ub = self.im_ub
            
        return np.array([image_lb, image_ub])
    
    def get_ranges(self, *args):
        """
            Computes the lower and upper bound images of the ImageStar
            
            return -> [image_lb : np.array([]) -> lower bound image,
                       image_ub : np.array([]) -> upper bound image]
        """
        solver = None
        options = None
        
        if len(args) == 0:
            solver = 'glpk'
            options = []
        elif len(args) == 1:
            solver = args[0]
            options = []
        elif len(args) == 2:
            solver = args[0]
            options = args[1]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 0, 1 or 2')
        
        self.validate_solver_name(solver)
        self.validate_options_list(options)
                
        image_lb = np.zeros((self.height, self.width, self.num_channels))
        image_ub = np.zeros((self.height, self.width, self.num_channels))

        size = self.height * self.width * self.num_channels
            
        if 'disp' in options:
            print(ESTIMATE_RANGE_STAGE_STARTED)
                
        for i in range(self.height):
            for j in range(self.width):
                for k in range(self.num_channels):
                    image_lb[i, j, k], image_ub[i, j, k] = self.get_range(i, j, k, solver, options)
                        
                    if 'disp' in options:
                        print(ESTIMATE_RANGE_STAGE_OVER)
                
        self.im_lb = image_lb
        self.im_ub = image_ub
            
        return np.array([image_lb, image_ub])
            
    def update_ranges(self, *args):
        """
            Updates local ranges for the MaxPooling operation
            
            points : np.array([*]) -> local points = [x1 y1 c1; x2 y2 c2; ...]
        """
        
        if len(args == 0):
            solver = 'glpk'
            options = []
        elif len(args == 1):
            solver = args[0]
            options = []
        elif len(args == 2):
            solver = args[0]
            options = args[1]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 0, 1 or 2')
        
        updated_ranges = []
        
        points = args[0]
        
        for i in range(len(points)):
            updated_ranges.append(self.get_range(points[i][0], points[i][1], points[i][2]), solver, options)
                                  
        return updated_ranges
            
        
            
    def get_num_attacked_pixels(self):
        """
            Computes the number of attacked pixels in the ImageStar
            
            return : int -> the number of pixels
        """
        
        V1 = np.zeros((self.height, self.width, self.num_channels))
        V3 = V1
        
        for i in range(1, self.numpred + 1):
            V2 = (self.V[:,:,:,i] != V1)
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
        
        start_points = None
        pool_size = None
        channel_id = None
        solver = None
        options = None
        
        if len(args) == 3:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = 'glpk'
            options = []
        elif len(args) == 4:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = args[3]
            options = []
        elif len(args) == 5:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = args[3]
            options = args[4]
        
        points = self.get_local_points(start_point, pool_size)
        points_num = len(points)
        
        if self.isempty(self.im_lb) or self.isempty(self.im_ub):
            image_lb, image_ub = self.get_ranges(solver, options)
        else:
            image_lb = self.im_lb
            image_ub = self.im_ub
        
        lb = image_lb[int(points[0,0]), int(points[0,1]), channel_id]
        ub = image_ub[int(points[0,0]), int(points[0,1]), channel_id]
        
        for i in range(1, points_num):
            if image_lb[int(points[i,0]), int(points[i,1]), channel_id] < lb:
                lb = image_lb[int(points[i,0]), int(points[i,1]), channel_id]
            
            if image_ub[int(points[i,0]), int(points[i,1]), channel_id] > ub:
                ub = image_ub[int(points[i,0]), int(points[i,1]), channel_id]
                
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
        
        assert (x0 >= 0 and y0 >= 0 and x0 + h - 1 < self.height \
                        and y0 + w - 1 < self.width), \
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
                
        return points
      
    def get_localMax_index(self, *args):
        """
            Gets local max index. Attempts to find the maximum point of the local image.
            It's used in over-approximate reachability analysis of the maxpooling operation
        
            startpoints : np.array([int, int]) -> startpoint of the local image
            pool_size : np.array([int, int]) -> the height and width of the max pooling layer
            channel_id : int -> the channel index
            
            return -> max_id
        """
        start_point = None
        pool_size = None
        channel_id = None
        solver = None
        options = None
        
        if len(args) == 3:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = 'glpk'
            options = []
        elif len(args) == 4:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = args[3]
            options = []
        elif len(args) == 5:
            start_point = args[0]
            pool_size = args[1]
            channel_id = args[2]
            solver = args[3]
            options = args[4]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 4, 5 or 6')
        
        self.validate_solver_name(solver)
        self.validate_options_list(options)
        
        points = self.get_local_points(start_point, pool_size)
    
        if self.isempty(self.im_lb) or self.isempty(self.im_ub):
            self.estimate_ranges()
    
        height = pool_size[0]
        width = pool_size[0]
        size = height * width
        
        lb = np.zeros((size, 1))
        ub = np.zeros((size, 1))
        
        for i in range(size):
            current_point = points[i, :].astype('int')
            
            lb[i] = self.im_lb[current_point[0], current_point[1], channel_id]
            ub[i] = self.im_ub[current_point[0], current_point[1], channel_id]
        
            
        [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
        
        a = np.argwhere((ub - max_lb_val) > 0)[:,0]
        a1 = np.argwhere((ub - max_lb_val) >= 0)[:,0]
        a = np.delete(a, np.argwhere(a==max_lb_id)[:,0])
        
        if self.isempty(a):
            max_id = np.array([np.append(points[max_lb_id, :].astype('int'), channel_id)])
        else:
            candidates = a1
            
            candidates_num = len(candidates)
            
            new_points = []
            new_points1 = np.zeros((candidates_num, 2))
            
            for i in range(candidates_num):
                selected_points = points[candidates[i], :].astype('int')
                new_points.append(np.append(selected_points, channel_id))
                new_points1[i, :] = selected_points
               
            self.update_ranges(new_points, solver, options)
            
            lb = np.zeros((candidates_num,1))
            ub = np.zeros((candidates_num,1))
            
            for i in range(candidates_num):
                #TODO: THIS SHOULD BE INITIALLY INT
                current_point = points[candidates[i], :]
                
                lb[i] = self.im_lb[int(current_point[0]), int(current_point[1]), int(channel_id)]
                ub[i] = self.im_ub[int(current_point[0]), int(current_point[1]), int(channel_id)]
                
            [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
            
            a = np.argwhere((ub - max_lb_val) > 0)[:,0]
            a = np.delete(a, np.argwhere(a==max_lb_id)[:,0])
            
            if self.isempty(a):
                max_id = new_points1[max_lb_val, :].astype('int')
            else:
                candidates1 = (ub - max_lb_val) >= 0
                max_id = new_points1[max_lb_id, :].astype('int')
                
                candidates1[candidates1 == max_lb_id] == []
                candidates_num = len(candidates1)
                
                max_id1 = np.append(max_id, channel_id)
                
                for j in range(candidates_num):
                    p1 = new_points[np.argwhere(candidates1 == True)[:,0][j]]
                    
                    if self.is_p1_larger_p2(np.array([p1[0], p1[1], channel_id]), \
                                            np.array([max_id[0], max_id[1], int(channel_id)]), \
                                            solver, options):
                        max_id1 = np.vstack((max_id1, p1))
                        
                        
                max_id = max_id1[1:max_id1.shape[0], :]
                        
                print('\nThe local image has %d max candidates' % max_id.shape[0])
                
        return max_id.astype('int')
          
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
            
            lb[i] = self.im_lb[current_point[0], current_point[1], channel_id]
            ub[i] = self.im_ub[current_point[0], current_point[1], channel_id]
        
            
        [max_lb_id, max_lb_val] = max(enumerate(lb), key=operator.itemgetter(1))
            
        max_id = np.argwhere((ub - max_lb_val) > 0)[:,0]
        
        return np.append(max_id, channel_id * np.zeros((len(max_id.shape)))).tolist()
    
    def update_max_id(self, name, max_id, pos):
        """
            Updates a matrix used for unmaxpooling reachability
        
            name : string -> name of the max pooling layer
            max_id : np.array([*int]) -> max indices
            pos : np.array([]) -> the position of the local pixel of the max map
                                  corresponding to the max_id
        """
        
        ids_num = len(self.max_ids)
        
        unk_layer_num = 0
        
        for i in range(ids_num):
            if self.max_ids[i].get_name() == name:
                self.max_ids[i].get_max_ids()[pos[0], pos[1], pos[2]] = max_id
                break
            else:
                unk_layer_num += 1
                
        if unk_laye_num == ids_num:
            raise Exception('error: %s' % ERRMSG_UNK_NAME_MAX_POOL_LAYER)
        
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
        
        if len(args == 2):
            p1 = args[0]
            p2 = args[1]
            solver = 'glpk'
            options = []
        elif len(args == 3):
            p1 = args[0]
            p2 = args[2]
            solver = args[3]
            options = []
        elif len(args == 4):
            p1 = args[0]
            p2 = args[1]
            solver = args[2]
            options = args[3]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 2, 3 or 4')
        
        C1 = np.zeros(self.numpred)
        
        for i in range(1, self.numpred + 1):
            C1[i-1] = self.V[p12[0], p12[1], p12[2], i] - \
                      self.V[p1[0], p1[1], p1[2], i]
                      
        d1 = self.V[p1[0], p1[1], p1[2], 0] - \
                 self.V[p12[0], p12[1], p12[2], 0]
                 
        new_C = np.vstack((self.C, C1))
        new_d = np.vstack((self.d, d1))
                
        if len(new_d.shape) == 2 and new_d.shape[1] == 1:
            new_d = np.reshape(new_d, (2,))
                
        f = np.zeros(self.numpred)
        
        if solver == 'linprog':
            if 'disp' in options:
                print('Optimising using Gurobi...')
            m = gp.Model()
            # prevent optimization information
            m.Params.LogToConsole = 0
            m.Params.OptimalityTol = 1e-9
            x = m.addMVar(shape=self.numpred, lb=self.pred_lb, ub=self.pred_ub)
            m.setObjective(f @ x, GRB.MINIMIZE)
            A = sp.csr_matrix(new_C)
            b = new_d
            m.addConstr(A @ x <= b)     
            m.optimize()
    
            if m.status == 2:   #feasible solution exist
                return True
            elif m.status == 3:
                return False
            else:
                raise Exception('error: exitflat = %d' % (m.status)) 
        elif solver == 'glpk':
            if 'disp' in options:
                print('Optimizing using GLPK...')
            #TODO: implement glpk here
            print('GLPK is not implemented for this method yet')
            
    @staticmethod
    def is_max(*args):
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
        
        if len(args == 4):
            max_map = args[0]
            ori_image = args[1]
            center = args[2]
            others = args[3]
            solver = 'glpk'
            options = []
        elif len(args == 5):
            max_map = args[0]
            ori_image = args[1]
            center = args[2]
            others = args[3]
            solver = args[4]
            options = []
        elif len(args == 6):
            max_map = args[0]
            ori_image = args[1]
            center = args[2]
            others = args[3]
            solver = args[4]
            options = args[5]
        else:
            raise Exception('\nInvalid number of input arguments. Should be 4, 5 or 6')
        
        size = others.shape[0]
        
        new_C = np.zeros((size, max_map.get_num_pred()))
        new_d = np.zeros((size, 1))
        
        for i in range(size):
            new_d[i] = ori_image.get_V()[center[0], center[1], center[2], 0] - \
                       ori_image.get_V()[others[i][0], others[i][1], others[i][2], 0]
        
            for j in range(max_map.get_num_pred()):
                new_C[i, j] = - ori_image.get_V()[center[0], center[1], center[2], j + 1] + \
                       ori_image.get_V()[others[i][0], others[i][1], others[i][2], j + 1]
        
        C1 = np.vstack((max_map.get_C(), new_C))
        d1 = np.vstack((max_map.get_d(), new_d))
        
        # TODO: remove redundant constraints here
        E = np.hstack((C1, d1))
        E = np.unique(E, axis = 0)
        
        new_C = E[:, 0:ori_image.get_num_pred()]
        new_d_gurobi = E[:, ori_image.get_num_pred()]
        new_d = np.reshape(new_d_gurobi, (new_d_gurobi.shape[0], 1))
                
        f = np.zeros(ori_image.get_num_pred())
        
        if solver == 'linprog':
            m = gp.Model()
            # prevent optimization information
            m.Params.LogToConsole = 0
            m.Params.OptimalityTol = 1e-9
            x = m.addMVar(shape=ori_image.get_num_pred(), lb=ori_image.get_pred_lb(), ub=ori_image.get_pred_ub())
            m.setObjective(f @ x, GRB.MINIMIZE)
            A = sp.csr_matrix(new_C)
            b = new_d_gurobi
            m.addConstr(A @ x <= b)     
            m.optimize()
    
            if m.status != 2:   #feasible solution exist
                 Exception('error: exitflat = %d' % (m.status))
            
            return new_C, new_d
        elif solver == 'glpk':
            if 'disp' in options:
                print('Optimizing using GLPK...')
            #TODO: implement glpk here
            print('GLPK is not implemented for this method yet')
    
    @staticmethod
    def reshape(input, new_shape):
        """
            Reshapes the ImageStar
            
            input : ImageStar -> the input ImageStar
            new_shape : np.array([]) -> new shape
            
            return -> a reshaped ImageStar
        """
        
        size = np.size(new_shape)
        
        assert size == 3, 'error: %s' % ERRMSG_INVALID_NEW_SHAPE
        
        assert np.prod(new_shape[:]) == input.get_height() * input.get_width() * input.get_num_channel(), \
               'error: %s' % ERRMSG_SHAPES_INCONSISTENCY
               
        new_V = np.reshape(input.get_V(), np.append(new_shape, input.get_num_pred() + 1))
        
        return ImageStar(new_V, input.get_C(), input.get_d(), \
                         input.get_pred_lb(), input.get_pred_ub(),
                         input.get_im_lb(), input.get_im_ub())
        
    @staticmethod
    def add_constraints(input, p1, p2):
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
        
        assert isinstance(input, ImageStar), 'error: %s' % ERRMSG_INPUT_NOT_IMAGESTAR
        
        new_d = input.get_V()[p2[0], p2[1], p2[2], 1] - input.get_V()[p2[0], p2[1], p2[2], 1]
        new_C = input.get_V()[p2[0], p2[1], p2[2], 1:input.get_num_pred() + 1]
        
        new_C = np.reshape(new_C, (1, input.get_num_pred()))
        
        new_C = np.vstack((input.get_C(), new_C))
        new_d = np.vstack((input.get_d(), new_d))
        
        return new_C, new_d
      
    def add_max_idx(self, name, id): 
        """
            Adds a max index. This is used for (un)maxpooling reachability
            
            name : string -> name of the max pooling layer
            id : int -> index
        """
      
        new_max_id = {
                'name' : name,
                'id' : id
            }
        
        self.max_ids.append(new_max_id)
        
    
    def add_input_size(self, name, input_size):
        """
            Adds a matrix used for unmaxpooling reachability
            
            name : string -> name of the max pooling layer
            input_size : np.array([*int]) -> input size of the original image
        """
         
        new_input_size = {
                'name' : name,
                'input_size' : id
            }
        
        self.max_input_sizes.append(new_input_size)
      
    def update_max_idx(self, name, max_id, pos):
        """
            Updates max index. This is used for (un)maxpooling reachability
            
            name : String -> name of the max pooling layer
            maxIdx : np.array([int*]) -> max indices
            pos : np.array([int*]) -> the position of the local pixel of the max map
            orresponding to the maxIdx
        """
        
        max_indices_size = len(self.max_ids)
        
        for i in range(max_indices_size):
            curr_index = self.max_ids[i]
            
            if curr_index['name'] == name:
                curr_index['id'][pos[0]][pos[1]][pos[2]] = max_id
                return
                
        raise Exception(ERRMSG_UNK_MP_LAYER_NAME)
      
##################### GET/SET METHODS #####################
        
    def get_V(self):
        """
            return -> the center and the basis matrix of the ImageStar
        """
        
        return self.V
        
    def get_C(self):
        """
            return -> the predicate of the ImageStar
        """
        
        return self.C
        
    def get_d(self):
        """
            return -> the free vector of the ImageStar
        """
        
        return self.d
        
    def get_pred_lb(self):
        """
            return -> the predicate lowerbound of the ImageStar
        """
        
        return self.pred_lb
        
    def get_pred_ub(self):
        """
            return -> the predicate upperbound of the ImageStar
        """
        
        return self.pred_ub
  
    def get_im_lb(self):
        """
            return -> the lowerbound image of the ImageStar
        """
        
        return self.im_lb
        
    def get_im_ub(self):
        """
            return -> the upperbound image of the ImageStar
        """
        
        return self.im_ub
    
    def get_IM(self):
        return self.im
    
    def get_LB(self):
        return self.lb
    
    def get_UB(self):
        return self.ub
    
    def get_height(self):
        """
            return -> the height of the ImageStar
        """
        
        return self.height
    
    def get_width(self):
        """
            return -> the width of the ImageStar
        """
        
        return self.width
    
    def get_num_channel(self):
        """
            return -> the channel of the ImageStar
        """
        
        return self.num_channels
    
    def get_num_pred(self):
        """
            return -> the channel of the ImageStar
        """
        
        return self.numpred
    
    def get_max_indices(self):
        """
            return -> max indices of the ImageStar
        """
        
        return self.max_ids
    
    def set_max_indices(self, new_max_indices):
        """
            sets max indices of the ImageStar
        """
        
        self.max_ids = new_max_indices
    
    def get_input_sizes(self):
        """
            return -> max input sizes of the ImageStar
        """
        
        return self.max_input_sizes
    
    def set_input_sizes(self, new_input_sizes):
        """
            sets max input sizes of the ImageStar
        """
        
        self.max_input_sizes = new_input_sizes
        
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
            
    def copy_deep(self, imagestar):            
        self.V = copy.deepcopy(imagestar.get_V())
        self.C = copy.deepcopy(imagestar.get_C())
        self.d = copy.deepcopy(imagestar.get_d())
        self.pred_lb = copy.deepcopy(imagestar.get_pred_lb())
        self.pred_ub = copy.deepcopy(imagestar.get_pred_ub())
        
        self.im_lb = copy.deepcopy(imagestar.get_im_lb())
        self.im_ub = copy.deepcopy(imagestar.get_im_ub())
        
        self.im = copy.deepcopy(imagestar.get_IM())
        self.lb = copy.deepcopy(imagestar.get_LB())
        self.ub = copy.deepcopy(imagestar.get_UB())
            
        self.height = imagestar.get_height()
        self.width = imagestar.get_width()
        self.num_channels = imagestar.get_num_channel()
            
        self.numpred = imagestar.get_num_pred()
            
        v_shape = self.V.shape
            
        if len(v_shape) == 3:
            self.V = np.reshape(self.V, (v_shape[0], v_shape[1], 1, v_shape[2]))
            self.num_channels = 1

    def validate_point_dim(self, point, height, width):
        return (point[0] > -1) and (point[0] <= self.height) and \
               (point[1] > -1) and (point[1] <= self.width) and \
               (point[2] > -1) and (point[2] <= self.num_channels)

    def validate_options_list(self, options):
        # TODO: this should be unified
        if not(type(options) == list):
            raise Exception(ERRMSG_INCONSISTENT_OPTIONS_LIST)
        
        if options == []:
            return
            
        available_options = ['disp']
        
        for option in options:
            if not (option in available_options):
                raise Exception(ERRMSG_INCONSISTENT_OPTIONS_LIST)
                
        
    
    def init_attributes(self):
        self.V = np.array([])
        self.C = np.array([])
        self.d = np.array([])
        
        self.im_lb = np.array([])
        self.im_ub = np.array([])
        
        self.im = np.array([])
        self.lb = np.array([])
        self.ub = np.array([])
    
        self.flatten_mode = 'F'
        
        self.max_ids = []
        self.max_input_sizes = []
    
    def validate_solver_name(self, name):
        assert (name == 'glpk' or name == 'linprog'), 'error: %s' % ERRMSG_INCONSISTENT_SOLVER_INPUT
           
    def is_scalar(self, param):
        return isinstance(param, np.ndarray)
    
    def is_scalar_attribute(self, attribute_id):
        return attribute_id in self.scalar_attributes_ids
    
    def get_attribute(self, i):
        return self.attributes[i]

    # def __str__(self):
    #     from zono import Zono
    #     print('class: %s' % self.__class__)
    #     print('height: %s \nwidth: %s \nnumChannel: %s' % (self.get_height(), self.get_width(), self.get_num_channel()))
    #     if self.im.size:
    #         print('IM: [shape: %s | type: %s]' % (self.im.shape, self.im.dtype))
    #     else:
    #         print('IM: []')
    #     if self.im_lb.size:
    #         print('LB: [shape: %s | type: %s]' % (self.im_lb.shape, self.im_lb.dtype))
    #     else:
    #         print('LB: []')
    #     if self.im_ub.size:
    #         print('UB: [shape: %s | type: %s]' % (self.im_ub.shape, self.im_ub.dtype))
    #     else:
    #         print('UB: []')       
    #     print('V: [shape: %s | type: %s]' % (self.get_V().shape, self.get_V().dtype))
    #     print('C: [shape: %s | type: %s]' % (self.get_C().shape, self.get_C().dtype))
    #     print('d: [shape: %s | type: %s]' % (self.get_d().shape, self.get_d().dtype))
    #     print('numPred: %s' % self.get_num_pred())
    #
    #     if self.get_pred_lb().size:
    #         print('predicate_lb: [shape: %s | type: %s]' % (self.get_pred_lb().shape, self.get_pred_lb().dtype))
    #     else:
    #         print('predicate_lb: []')
    #
    #     if self.get_pred_ub().size:
    #         print('predicate_ub: [shape: %s | type: %s]' % (self.get_pred_ub().shape, self.get_pred_ub().dtype))
    #     else:
    #         print('predicate_ub: []')
    #
    #     if self.get_im_lb().size:
    #         print('im_lb: [shape: %s | type: %s]' % (self.get_im_lb().shape, self.get_im_lb().dtype))
    #     else:
    #         print('im_lb: []')
    #
    #     if self.get_im_ub().size:
    #         print('im_ub: [shape: %s | type: %s]' % (self.get_im_ub().shape, self.get_im_ub().dtype))
    #     else:
    #         print('im_ub: []')
    #
    #     if self.max_ids.size:
    #         print('MaxIdxs: [shape: %s | type: %s]' % (self.max_ids.shape, self.max_ids.dtype))
    #     else:
    #         print('MaxIdxs: []')
    #
    #     if self.max_input_sizes.size:
    #         print('InputSizes: [shape: %s | type: %s]' % (self.max_input_sizes.shape, self.max_input_sizes.dtype))
    #     else:
    #         print('InputSizes: []')     
    #     return '\n'
    #
    # def __repr__(self, mat_ver=True):
    #     """
    #         mat_ver :   1 -> print images in [[[height, width] x channel] x (numPred+1)]
    #                     0 -> print images in [height x [width x [channel, (numPred+1)]]]
    #     """
    #     if mat_ver == False:
    #         return "class: %s \nheight: %s\nwidth: %s\nnumChannels: %s\nIM: \n%s\nLB: \n%s\nUB: \n%s\nV: \n%s \nC: \n%s \nd: %s\nnumPred: %s\npredicate_lb: %s \npredicate_ub: %s\nim_lb: \n%s\nimb_ub: \n%s\nMaxIdxs: %s\nInputSizes: %s" % \
    #             (self.__class__, self.get_height(), self.get_width(), self.get_num_channel(), self.im, self.im_lb, self.im_ub, self.get_V(), self.get_C(), self.get_d(), self.get_num_pred(), self.get_pred_lb(), self.get_pred_ub(), self.get_im_lb(), self.get_im_ub(),self.max_ids,self.max_input_sizes)
    #     else:
    #         print("class: %s \nheight: %s\nwidth: %s\nnumChannels: %s" % (self.__class__, self.get_height(), self.get_width(), self.get_num_channel()))
    #         if self.im.size:
    #             print("IM: \n%s\n" % self.im.transpose([-1, 0, 1]))
    #         else:
    #             print('IM: []')
    #         if self.im_lb.size:
    #             print("LB: \n%s\n" % self.im_lb.transpose([-1, 0, 1]))
    #         else:
    #             print('LB: []')
    #         if self.im_ub.size:
    #             print("UB: \n%s\n" % self.im_ub.transpose([-1, 0, 1]))
    #         else:
    #             print('UB: []')
    #         print("V: \n%s \nC: \n%s \nd: %s \nnumPred: %s\npredicate_lb: %s \npredicate_ub: %s" % (self.get_V().transpose([3,2,0,1]), self.get_C(), self.get_d(), self.get_num_pred(), self.get_pred_lb(), self.get_pred_ub()))
    #         if self.get_im_lb().size:
    #             print("im_lb: \n%s\n" % self.get_im_lb().transpose([-1, 0, 1]))
    #         else:
    #             print("im_ub: []")
    #         if self.get_im_ub().size:
    #             print("im_ub: \n%s" % self.get_im_ub().transpose([-1, 0, 1]))
    #         else:
    #             print("im_ub: []")
    #         print("MaxIdxs: %s\nInputSizes: %s" % (self.max_ids, self.max_input_sizes))
    #         return "" 