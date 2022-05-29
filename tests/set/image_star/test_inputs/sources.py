CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT = 0
CONSTRUCTOR_BOUNDS_INIT = 1
EVALUATION_INIT = 2
AFFINEMAP_INIT = 3
TO_STAR_INIT = 4
IS_EMPTY_INIT = 5
CONTAINS_INIT = 6
PROJECT2D_INIT = 7
GETRANGE_INIT = 8
ESTIMATE_RANGE_INIT = 9
ESTIMATE_RANGES_INIT = 10
GET_RANGES_INIT = 11
UPDATE_RANGES_INIT = 12
GET_NUM_ATTACK_PIXELS_INIT = 13
GET_LOCAL_POINTS_INIT = 14
GET_LOCAL_BOUND_INIT = 15
GET_LOCAL_MAX_INDEX_INIT = 16
GET_LOCAL_MAX_INDEX2_INIT = 17
IS_P1_LARGER_P2_INIT = 18

TEST_LB_ID = 0
TEST_UB_ID = 1

EVAL_INPUT_ID = 0
EVAL_OUTPUT_ID = 1

V_ID = 0
C_ID = 1
D_ID = 2
PREDICATE_LB_ID = 3
PREDICATE_UB_ID = 4

SCALE_ID = 5
OFFSET_ID = 6
AFFINEMAP_OUTPUT_ID = 7

TRUE_INPUT_ID = 5

INPUT_ID = 5
ESTIMATE_RANGE_OUTPUT_ID = 6

ESTIMATE_RANGES_OUTPUT_ID = 5

OUTPUT_ID = 5

INPUT_ID = 5
OUTPUT_ID = 6

V_CANDIDATES_ID = 7
C_CANDIDATES_ID = 8
D_CANDIDATES_ID = 9
PREDICATE_LB_CANDIDATES_ID = 10
PREDICATE_UB_CANDIDATES_ID = 11
IM_LB_CANDIDATES_ID = 12
IM_UB_CANDIDATES_ID = 13

INPUT_CANDIDATES_ID = 14
OUTPUT_CANDIDATES_ID = 15


import numpy as np
import mat73

sources = {
        CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
            ],
        CONSTRUCTOR_BOUNDS_INIT : [
                "test_inputs/fmnist_img/test_lb.mat",
                "test_inputs/fmnist_img/test_ub.mat"
            ],
        EVALUATION_INIT : [
                "test_inputs/test_eval_input.mat",
                "test_inputs/test_eval_output.mat"
            ],
        AFFINEMAP_INIT : [
                "test_inputs/imgstar_input/test_V.mat",
                "test_inputs/imgstar_input/test_C.mat",
                "test_inputs/imgstar_input/test_d.mat",
                "test_inputs/imgstar_input/test_predicate_lb.mat",
                "test_inputs/imgstar_input/test_predicate_ub.mat",
                
                "test_inputs/test_affineMap_scale.mat",
                "test_inputs/test_affineMap_offset.mat",
                
                "test_inputs/test_affineMap_output.mat"
            ],
        TO_STAR_INIT : [
                "test_inputs/fmnist_img/test_lb.mat",
                "test_inputs/fmnist_img/test_ub.mat"
            ],
        IS_EMPTY_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat"
            ],
        CONTAINS_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_contains_true_input.mat"
            ],
        PROJECT2D_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_point1.mat",
                "test_inputs/test_point2.mat"
            ],
        GETRANGE_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_range_input.mat",
                "test_inputs/test_get_range_output.mat"
            ],
        ESTIMATE_RANGE_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_estimate_range_input.mat",
                "test_inputs/test_estimate_range_output.mat"
            ],
        ESTIMATE_RANGES_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_estimate_ranges_output.mat"
            ],
        GET_RANGES_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_ranges_output.mat"
            ],
        UPDATE_RANGES_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_update_ranges_output.mat"
            ],
        GET_NUM_ATTACK_PIXELS_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_num_attacked_pixels_output.mat"
            ],
        GET_LOCAL_POINTS_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_local_points_input.mat",
                "test_inputs/test_get_local_points_output.mat"
            ],
        GET_LOCAL_BOUND_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_local_bound_input.mat",
                "test_inputs/test_get_local_bound_output.mat"
            ],
        GET_LOCAL_MAX_INDEX_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_local_max_index_input.mat",
                "test_inputs/test_get_local_max_index_output.mat",
                
                "test_inputs/imgstar_local_max_index/test_V.mat",
                "test_inputs/imgstar_local_max_index/test_C.mat",
                "test_inputs/imgstar_local_max_index/test_d.mat",
                "test_inputs/imgstar_local_max_index/test_predicate_lb.mat",
                "test_inputs/imgstar_local_max_index/test_predicate_ub.mat",
                "test_inputs/imgstar_local_max_index/test_im_lb.mat",
                "test_inputs/imgstar_local_max_index/test_im_ub.mat",
                
                "test_inputs/test_get_local_max_index_input_candidates.mat",
                "test_inputs/test_get_local_max_index_output_candidates.mat"
            ],
        GET_LOCAL_MAX_INDEX2_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_get_local_max_index2_input.mat",
                "test_inputs/test_get_local_max_index2_output.mat",
            ],
        IS_P1_LARGER_P2_INIT : [
                "test_inputs/fmnist_img/test_V.mat",
                "test_inputs/fmnist_img/test_C.mat",
                "test_inputs/fmnist_img/test_d.mat",
                "test_inputs/fmnist_img/test_predicate_lb.mat",
                "test_inputs/fmnist_img/test_predicate_ub.mat",
                
                "test_inputs/test_is_p1_larger_p2_input.mat",
                "test_inputs/test_is_p1_larger_p2_output.mat"
                ]
    }


########################## UTILS ##########################
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])
    