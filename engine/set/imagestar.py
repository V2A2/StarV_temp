#!/usr/bin/python3
import numpy as np

class ImageStar:
    # Class for representing set of images using Star set
    # An image can be attacked by bounded noise. An attacked image can
    # be represented using an ImageStar Set
    # author: Sung Woo Choi
    # date: 9/30/2021

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

    # constructor using 2D representation / 1D representation of an ImageStar
    def __init__(obj, V = np.array([]),     # a cell (size = numPred)
                C = np.array([]),           # a constraints matrix of the predicate
                d = np.array([]),           # a constraints vector of the predicate
                pred_lb = np.array([]),     # lower bound vector of the predicate
                pred_ub = np.array([]),     # upper bound vector of the predicate
                im_lb = np.array([]),       # lower bound image of the ImageStar
                im_ub = np.array([]),       # upper bound image of the ImageStar
                IM = np.array([]),          # center image (high-dimensional array)
                LB = np.array([]),          # lower bound of attack (high-dimensional array)
                UB = np.array([])):         # upper bound of attack (high-dimensional array)
        from engine.set.star import Star

        obj.MaxIdxs = np.array([])           # used for unmaxpooling operation in Segmentation network
        obj.InputSizes = np.array([])      # used for unmaxpooling operation in Segmentation network
        # The 2D representation of ImageStar is convenient for reachability analysis
        if V.size and C.size and d.size and pred_lb.size and pred_ub.size:
            assert C.shape[0] == d.shape[0], 'error: inconsistent dimension between constraint matrix and constraint vector'
            assert d.shape[1] == 1, 'error: invalid constraint vector, vector should have one column'
            
            obj.numPred = C.shape[1]
            obj.C = C
            obj.d = d

            if im_lb.size and im_ub.size:
                assert C.size[1] == pred_ub.size[0] == pred_lb.size[0], 'error: number of predicates is different from the size of the lower bound or upper bound predicate vector'
                assert pred_lb[1] == pred_ub[1] == 1, 'error: invalid lower/upper bound predicate vector, vector should have one column'

                obj.pred_lb = pred_lb
                obj.pred_ub = pred_ub

                n = V.shape
                if len(n) == 3:
                    obj.V = V
                    [obj.numChannel, obj.height, obj.width] = n
                elif len(n) == 4:
                    assert n[0] == obj.numPred + 1, 'error: inconsistency between the basis matrix and the number of predicate variables'
                    obj.V = V
                    [_, obj.numChannel, obj.height, obj.width] = n
                elif len(n) == 2:
                    obj.V = V
                    obj.numChannel = 1
                    [obj.height, obj.width] = n
                else:
                    raise Exception('invalid basis matrix')

                if im_lb.size > 0 and (im_lb.shape[0] != obj.height or im_lb.shape[1] != obj.width):
                    raise Exception('error: inconsistent dimension between lower bound image and the constructed imagestar')
                else:
                    obj.im_lb = im_lb
                    
                if im_ub.size > 0 and (im_ub.shape[0] != obj.height or im_ub.shape[1] != obj.width):
                    raise Exception('error: inconsistent dimension between upper bound image and the constructed imagestar')
                else:
                    obj.im_ub = im_ub
                return

        if IM.size and LB.size and UB.size:
            # input center image and lower and upper bound matrices (box-representation)
            n = IM.shape # n[0] is number of channels 
                         # n[1] and n[2] are height and width of image
            l = LB.shape
            u = UB.shape

            assert n[0] == l[0] == u[0] and n[1] == l[1] == u[1], 'error: inconsistency between center image and attack bound matrices'
            assert len[n] == len[l] == len[u], 'error: inconsistency between center image and attack bound matrices'

            if len(n) == len(l) == len(u) == 2:
                obj.numChannel = 1
                [obj.height, obj.width] = n
                obj.IM = IM
                obj.LB = LB
                obj.UB = UB
            elif len[n] == len[l] == len[u] == 3:
                if n[3] == l[3] == u[3]:
                    [obj.numChannel, obj.height, obj.width] = n
                    obj.IM = IM
                    obj.LB = LB
                    obj.UB = UB
                else:
                    raise Exception('error: inconsistent number of channels between the center image and the bound matrices')
            else:
                raise Exception('error: inconsistent number of channels between the center image and the bound matrices')

            obj.im_lb = IM + LB # lower bound image
            obj.im_ub = IM + UB # upper bound image

            # converting box ImageStar to an array of 2D Stars
            n = obj.im_lb.shape
            if len(3) == 3:
                I = Star(obj.im_lb.flatten(), obj.im_ub.flatten())
                obj.V = np.reshape(I.V, (I.nVar + 1, n[0], n[1], n[2]))
            else:
                I = Star(obj.im_lb.flatten(), obj.im_ub.flatten())
                obj.V = np.reshape(I.V, (I.nVar + 1, n[0], n[1]))

            obj.C = I.C
            obj.d = I.d
            obj.pred_lb = I.predicate_lb
            obj.pred_ub = I.predicate_ub
            obj.numPred = I.nVar
            return            
                
        raise Exception('error: failed to create ImageStar')

#------------------check if this function is working--------------------------------------------
    # first implemnt Star.contains()
    # randomly generate a set of images from an imagestar set
    # def sample(obj, N):
    #     # @N: number of images
    #     from engine.set.star import Star
    #     assert obj.V.size, 'error: the ImageStar is an empty set'
        
    #     if obj.C.size == 0 or obj.d.size == 0:
    #         images = obj.IM
    #     else:
    #         V = np.hstack((np.zerons(obj.numPred, 1), np.eye(obj.numPred)))
    #         S = Star(V, obj.C, obj.d)
    #         pred_samples = S.sample(N)

#------------------check if this function is working--------------------------------------------
    def evaluate(obj, pred_val):
        assert obj.V.size, 'error: the imagestar is an empty set'
        assert pred_val.shape[1] == 1, 'error: invalid predicate vector'
        assert pred_val.shape[0] == obj.numPred, 'error: inconsistency between the size of the predicate vector and the number of predicates in the ImageStar'

        image = np.zeros(obj.numChannel, obj.height, obj.width)
        for i in range(obj.numChannel):
            image[i, :, :] = obj.V[1, i, :, :]
            for j in range(1, obj.numPred + 1):
                image[i, :, :] = image[i, :, :] + pred_val[j-1] * obj.V[j, i, :, :]
        return image
    
 #------------------check if this function is working--------------------------------------------
    # affineMap of an ImageStar is another ImageStar
    # y = scale * x + offset
    def affineMap(obj, scale, offset):
        # @scale: scale coefficient [NumChannels x 1 x 1] array
        # @offset: offset coeeficient [NumChannels x 1 x 1] array
        # return: a new ImageStar

        assert scale.size and not np.isscalar(scale) and scale.shape[0] == obj.numChannels, 'error: inconsistent number of channels between scale array and the ImageStar'
        
        if scale.size:
            new_V = scale * obj.V
        else:
            new_V = obj.V

        if offset.size:
            new_V[1, :, :, :] = new_V[1, :, :, :] + offset
        
        return ImageStar(new_V, obj.C, obj.d, obj.pred_lb, obj.pred_ub)

#------------------check if this function is working--------------------------------------------
    # transform to Star
    def toStar(obj):
        from engine.set.star import Star
        nc = obj.numChannel
        h = obj.height
        w = obj.width
        np = obj.numPred

        N = h*w*nc # total number of pixels in the input image
        V1 = np.zeros(N, np+1)
        for j in range(np+1):
            V1[:, j] = obj.V[j, :,:,:].reshape(N, 1)
        
        if obj.im_lb.size and obj.im_ub.size:
            state_lb = obj.im_lb.reshape(-1,1)
            state_ub = obj.im_ub.reshape(-1,1)
            return Star(V1, obj.C, obj.d, obj.pred_lb. obj.pred_ub, state_lb, state_ub)
        else:
            return Star(V1, obj.C, obj.d, obj.pred_lb, obj.pred_ub)
        
    # checking if an ImageStar is an empty set
    def isEmptySet(obj):
        S = obj.toStar()
        return S.isEmptySet()

    # contain, check if ImageStar contains an image
    # def contains(obj, image):

    # projection of ImageStar on specific 2d plane
    # def project2D(obj, point1, point2):

    # get ranges of a state at specific position
    # def getRange(obj, vert_ind, horiz_ind, chan_ind, lp_solver = 'gurobi'):

    # get lower bound and upper bound images of an ImageStar
    # def getRanges(obj, lp_solver = 'gurobi'):

    # estimate range quickly using only predicate bound information
    # def estimateRange(obj, h, w, c):
    #     # @h: height index
    #     # @w: width index
    #     # @c: channel index
    #     # return: xmin: min of x[h, w, c]
    #     #         xmax: max of x[h, w, c]

    # estimate ranges quickly using only predicate bound information
    # def estimateRanges(obj, disp_opt = ''):

    # update local ranges for Max Pooling operation
    # def updateRanges(obj, points, lp_solver = 'gurobi'):

    # estimate the number of attacked pixels
    # def getNumAttackedPixels(obj):
        # return: number of attacked pixels in an ImageStar

    # get local bound for Max Pooling operation
    # def get_localBound(obj, PoolSize, channel_id, lp_solver = 'gurobi'):

    # get all local points index for Max Pooling operation
    # def get_localPoints(obj, startpoint, PoolSize):
        # @startpoint: startpoint of the local(partial) image
        #              startpoint = [x1 y1]
        # @PoolSize: [height width] the height and width of max pooling layer
        # @points: all index of all points for a single max
        # pooling oepration (including the startpoint)

    # get local max index, this method tries to find thee maximum point
    # of a local image, used in over-approximate reachability analysis
    # of maxpooling operation
    # def get_localMax_index(obj, startpoint, PoolSize, channel_id, lp_solver = 'gurobi'):

    # get local max index, this method tries to find the maximum point 
    # of a local image, used in over-approximate reachability analysis
    # of maxpooling operation
    # def get_localMax_index2(obj, startpoint, PoolSize, channel_id):
        # @startpoint: startpoint of the local(partial) image
        #              startpoint = [x1 y1]
        # @PoolSize: [height width] the height and width of max pooling layer
        # @channel_id: the channel index
        # @max_id: = []: we don't know which one has maximum value,
        #   i.e., the maximum values may be the intersection between of several pixel values.
        #           = [xi yi]: the point that has maximum value

    # add maxidx used for unmaxpooling reachability
    # def addMaxIdx(obj, name, maxIdx):
        # @name: name of the max pooling layer
        # @maxIdx: max indexes











