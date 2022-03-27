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
OUTPUT_ID = 6

OUTPUT_ID = 5

OUTPUT_ID = 5

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
            ]
    }


########################## UTILS ##########################
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])
    