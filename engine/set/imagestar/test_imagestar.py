import unittest

from test_inputs.sources import *
from imagestar import ImageStar

class TestImageStar(unittest.TestCase):
    """
        Tests the ImageStar class
    """

    def test_add_constraints_basic(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> new constraints
        """
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_C, test_d = ImageStar.add_constraints(test_star, [4, 4, 0], [1,1,0])
        
    def test_affine_mapping(self):
        """
            Test affine mapping -> ImageStar .* scale + offset
        
            scale : float -> affine map scale
            offset : np.array([*float]) -> affine map offset
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
                
        test_am_output = read_csv_data(sources[AFFINEMAP_INIT][AFFINEMAP_OUTPUT_ID])
                
        test_V = np.reshape(read_csv_data(sources[AFFINEMAP_INIT][V_ID]), (1, 1, 72, 1007))
        test_C = read_csv_data(sources[AFFINEMAP_INIT][C_ID])
        test_d = read_csv_data(sources[AFFINEMAP_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[AFFINEMAP_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[AFFINEMAP_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_scale = read_csv_data(sources[AFFINEMAP_INIT][SCALE_ID])
        test_offset = read_csv_data(sources[AFFINEMAP_INIT][OFFSET_ID])
        
        am_result = test_star.affine_map(test_scale, test_offset)
        
        self.assertEqual(am_result.get_V().all(), test_am_output.all())

    def test_constructor_predicate_boundaries_init(self):
        """
            Tests the initialization with:
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
    
        completion_flag = True
    
        test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
    
        try:
            test_star = ImageStar(
                    test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
                )
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)
            
        
    def test_constructor_image_init(self):
        """
            Tests the initialization with:
            IM -> ImageStar
            LB -> Lower image
            UB -> Upper image
        """
        completion_flag = True
        
        test_IM = np.zeros((4, 4, 3))
        test_LB = np.zeros((4, 4, 3))
        test_UB = np.zeros((4, 4, 3))
    
    
        test_IM[:,:,0] = np.array([[1, 1, 0, 1], [0, 0, 1, 1], [1, 0, 1, 0], [1, 1, 1, 1]])
        test_IM[:,:,1] = np.array([[0, 1, 0, 0], [1, 0, 0, 1], [0, 1, 1, 0], [0, 0, 0, 1]])
        test_IM[:,:,2] = np.array([[1, 1, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0], [1, 0, 1, 0]])
    
        test_LB[:,:,0] = np.array([[-0.1, -0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]) # attack on pixel (1,,1,) and (1,,2)
        test_LB[:,:,1] = np.array([[-0.1, -0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_LB[:,:,2] = test_LB[:,:,1]
    
        test_UB[:,:,0] = np.array([[0.1, 0.2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,1] = np.array([[0.1, 0.15, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        test_UB[:,:,2] = test_UB[:,:,1]
    
        try:
            test_star = ImageStar(
                    test_IM, test_LB, test_UB
                )
        except Exception as ex:
            process_exception(ex)
            completion_flag = False
            
        self.assertEqual(completion_flag, True)
        
    def test_constructor_bounds_init(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """
        completion_flag = True
        
        test_lb = read_csv_data(sources[CONSTRUCTOR_BOUNDS_INIT][TEST_LB_ID])
        test_ub = read_csv_data(sources[CONSTRUCTOR_BOUNDS_INIT][TEST_UB_ID])
        
    
        try:
            test_star = ImageStar(
                    test_lb, test_ub
                )
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)

    def test_containts_false(self):
        """
            Checks if the initialized ImageStar contains the given image
                       
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_input -> the input image
        """
        
        test_V = np.reshape(read_csv_data(sources[CONTAINS_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONTAINS_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[CONTAINS_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[CONTAINS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONTAINS_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_input = read_csv_data(sources[CONTAINS_INIT][TRUE_INPUT_ID])
        
        test_result = True

        try:
            test_result = test_star.contains(test_input)
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(test_result, False)
        
    def test_estimate_range_valid_input(self):
        """
            Tests the ImageStar's range estimation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_input -> valid input range
            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[ESTIMATE_RANGE_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[ESTIMATE_RANGE_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[ESTIMATE_RANGE_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[ESTIMATE_RANGE_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[ESTIMATE_RANGE_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        range_input = np.array(read_csv_data(sources[ESTIMATE_RANGE_INIT][INPUT_ID]))
        range_output = np.array([read_csv_data(sources[ESTIMATE_RANGE_INIT][ESTIMATE_RANGE_OUTPUT_ID])])
        
        self.assertEqual(test_star.estimate_range(range_input[VERT_ID], range_input[HORIZ_ID], range_input[CHANNEL_ID]).all(), range_output.all())

    def test_evaluation_valid_input(self):
        """
            Tests evaluation using predicate initialization
        
            eval_input : int -> number of images
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
                
        test_eval_input = read_csv_data(sources[EVALUATION_INIT][EVAL_INPUT_ID])
        test_eval_output = read_csv_data(sources[EVALUATION_INIT][EVAL_OUTPUT_ID])
                
        test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
                
        try:
            test_result = test_star.evaluate(test_eval_input)
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

    def test_get_local_bound_default(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4]), test_bounds_output)

                
    def test_get_local_bound_glpk(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'glpk'), test_bounds_output)

    def test_get_local_bound_gurobi(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'linprog'), test_bounds_output)

    def test_get_local_bound_glpk_disp(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'glpk', ['disp']), test_bounds_output)

    def test_get_local_bound_gurobi_disp(self):
        """
            Tests the ImageStar's method that calculates the local bounds for the given point and pool size
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_BOUND_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_BOUND_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_BOUND_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_bounds_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_BOUND_INIT][INPUT_ID])])
        test_bounds_output = read_csv_data(sources[GET_LOCAL_BOUND_INIT][OUTPUT_ID]).tolist()
                                
        self.assertEqual(test_star.get_local_bound(test_bounds_input[0:2], test_bounds_input[2:4], test_bounds_input[4], 'linprog', ['disp']), test_bounds_output)


    def test_get_local_max_index_empty_candidates(self):
        """
            Tests the ImageStar's method that calculates the local maximum point of the local image.
            Candidates set will be empty 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> valid input
            local_index -> valid output bounds
        """
    
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_ID])  - 1).astype('int').tolist()
    
        test_result = test_star.get_localMax_index(test_local_index_input[0:2] - 1, test_local_index_input[2:4], test_local_index_input[4] - 1, 'linprog')
    
    
        self.assertEqual((test_result == test_local_index_output).all(), True)

    def test_get_local_max_index_candidates(self):
        """
            Tests the ImageStar's method that calculates the local maximum point of the local image. 
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            local_index -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][V_CANDIDATES_ID]), (24, 24, 3, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][C_CANDIDATES_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][D_CANDIDATES_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_LB_CANDIDATES_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][PREDICATE_UB_CANDIDATES_ID])
        test_im_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_LB_CANDIDATES_ID])
        test_im_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][IM_UB_CANDIDATES_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
               
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][INPUT_CANDIDATES_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX_INIT][OUTPUT_CANDIDATES_ID])).tolist()
                        
        test_result = test_star.get_localMax_index(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4])
                                
        self.assertEqual(test_result, test_local_index_output)
        
    def test_get_local_points(self):
        """
            Tests the ImageStar's method that calculates the local points for the given point and pool size

            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            bounds_output -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_POINTS_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_POINTS_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_POINTS_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_POINTS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_POINTS_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        test_points_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_POINTS_INIT][INPUT_ID])])
        test_points_output = np.array(read_csv_data(sources[GET_LOCAL_POINTS_INIT][OUTPUT_ID]))
                                
        test_result = test_star.get_local_points(test_points_input[0:2], test_points_input[2:4])
                                
        self.assertEqual(test_result.all(), test_points_output.all())

    def test_get_local_max_index2_candidates(self):
        """
            Tests the ImageStar's method that calculates the local maximum point of the local image. 
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            input -> valid input
            local_index -> valid output bounds
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][V_ID]), (24, 24, 3, 785))
        test_C = np.reshape(read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][PREDICATE_UB_ID])
        test_im_lb = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][IM_LB_ID])
        test_im_ub = read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][IM_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub, test_im_lb, test_im_ub
            )
               
        test_local_index_input = np.array([int(item) for item in read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][INPUT_LOCAL_MAX_INDEX2_ID])])
        test_local_index_output = (read_csv_data(sources[GET_LOCAL_MAX_INDEX2_INIT][OUTPUT_LOCAL_MAX_INDEX2_ID])).tolist()
                         
        test_result = test_star.get_localMax_index2(test_local_index_input[0:2], test_local_index_input[2:4], test_local_index_input[4])
                                
        self.assertEqual(test_result, test_local_index_output)
        
    def test_num_attacked_pixels(self):
        """
            Tests the ImageStar's method that calculates the number of attacked pixels
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
               
        attacked_pixels_num_output = np.array([read_csv_data(sources[GET_NUM_ATTACK_PIXELS_INIT][ATTACKPIXNUM_OUTPUT_ID])])
                                
        test_result = test_star.get_num_attacked_pixels()
                                
        self.assertEqual(test_result, attacked_pixels_num_output)

    def test_contains_true(self):
        """
            Tests the ImageStar's range method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            range_input -> input
            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[GETRANGE_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[GETRANGE_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GETRANGE_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GETRANGE_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GETRANGE_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        range_input = np.array([int(item) for item in read_csv_data(sources[GETRANGE_INIT][INPUT_ID])])
        range_output = np.array([read_csv_data(sources[GETRANGE_INIT][OUTPUT_ID])])
        
        test_result = test_star.get_range(*range_input)
        
        self.assertEqual(test_result.all(), range_output.all())

    def test_get_ranges(self):
        """
            Tests the ImageStar's ranges calculation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[GET_RANGES_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[GET_RANGES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[GET_RANGES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[GET_RANGES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[GET_RANGES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        ranges_output = np.array([read_csv_data(sources[GET_RANGES_INIT][GETRANGES_OUTPUT_ID])])
        
        test_result = test_star.get_ranges()
        
        self.assertEqual(test_result.all(), ranges_output.all())

    def test_is_empty_false(self):
        """
            Checks if the initialized ImageStar is empty
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
        
        test_V = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[IS_EMPTY_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[IS_EMPTY_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_EMPTY_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        completion_flag = True
        
        try:
            test_result = test_star.is_empty_set()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(completion_flag, True)

    def test_is_empty_true(self):
        """
            Checks if the empty ImageStar is empty
        """
        
        test_star = ImageStar()
        
        completion_flag = True
        
        try:
            test_result = test_star.is_empty_set()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(completion_flag, True)
        
    def test_contains_true(self):
        """
            Tests the ImageStar's projection on the given plain formulated by two points
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_point1 -> the first point
            test_point2 -> the second point
        """
        
        test_V = np.reshape(read_csv_data(sources[PROJECT2D_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[PROJECT2D_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[PROJECT2D_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[PROJECT2D_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_point1 = np.array([4,4,0])#read_csv_data(sources[PROJECT2D_INIT][POINT1_ID])
        test_point2 = np.array([3,6,0])#read_csv_data(sources[PROJECT2D_INIT][POINT2_ID])
        
        completion_flag = True
        
        try:
            test_result = test_star.project2D(test_point1, test_point2)
        except Exception as ex:
            process_exception(ex)
            completion_flag = False
            
        self.assertEqual(completion_flag, True)
        
    def test_reshape(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> new shape
        """
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_result = ImageStar.reshape(test_star, [28, 14, 2])

    def test_sampling(self):
        """
            Tests sampling using predicate initialization
        
            N : int -> number of images
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
        """
        
        test_N = 2
        
        test_V = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][C_ID]), (1, 784))
        test_d = np.array([read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][D_ID])])
        test_predicate_lb = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        try:
            images = test_star.sample(test_N)
            
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

        self.assertEqual(completion_flag, True)
        
    def test_basic_to_star(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """
        
        test_lb = read_csv_data(sources[TO_STAR_INIT][TEST_LB_ID])
        test_ub = read_csv_data(sources[TO_STAR_INIT][TEST_UB_ID])
    
        completion_flag = True
    
        try:
            test_star = ImageStar(
                    test_lb, test_ub
                )
            
            converted = test_star.to_star()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)

    def test_update_ranges(self):
        """
            Tests the ImageStar's ranges calculation method
            
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound

            range_output -> valid output range
        """
        
        test_V = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[UPDATE_RANGES_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[UPDATE_RANGES_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[UPDATE_RANGES_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        ranges_input = [np.array([0, 0, 0])]
        
        ranges_output = np.array([read_csv_data(sources[UPDATE_RANGES_INIT][UPDATERANGES_OUTPUT_ID])])
                
        ranges = test_star.update_ranges(ranges_input)
                
        res_flag = True
        
        for i in range(len(ranges)):
            if ranges[i].all() != ranges_output[i].all():
                res_flag = False
                break
                
        self.assertEqual(res_flag, True)
        
    def test_is_max(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> valid input
            local_index -> valid output bounds
        """
        raise NotImplementedError
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
    
        test_points_input = np.array([int(item) for item in read_csv_data(sources[IS_P1_LARGER_P2_INIT][INPUT_ID])])
        test_points_output = (read_csv_data(sources[IS_P1_LARGER_P2_INIT][OUTPUT_ID])  - 1).tolist()
    
        completion_flag = True
        
        try:
            test_result = test_star.is_p1_larger_p2(test_points_input[0:3], test_points_input[3:6])
    
            self.assertEqual(test_result, test_points_output)
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
            
        self.assertEqual(completion_flag, True)
        
    def test_is_p1_larger_p2(self):
        """
            Tests the ImageStar's method that compares two points of the ImageStar 
    
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
    
            input -> valid input
            local_index -> valid output bounds
        """
    
        #raise NotImplementedError
    
        test_V = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][V_ID]), (28, 28, 1, 785))
        test_C = np.reshape(read_csv_data(sources[IS_P1_LARGER_P2_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[IS_P1_LARGER_P2_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[IS_P1_LARGER_P2_INIT][PREDICATE_UB_ID])
    
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
            
        test_points_input = np.array([int(item) for item in read_csv_data(sources[IS_P1_LARGER_P2_INIT][INPUT_ID])])
        test_result = test_star.is_p1_larger_p2(test_points_input[0:3], test_points_input[3:6])

        self.assertEqual(test_result, True)


if __name__ == '__main__':
    unittest.main()
