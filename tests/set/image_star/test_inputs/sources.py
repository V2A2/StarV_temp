CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT = 0
CONSTRUCTOR_BOUNDS_INIT = 1
EVALUATION_INIT = 2
AFFINEMAP_INIT = 3

LB_ID = 0
UB_ID = 1

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
            ]
    }


########################## UTILS ##########################
def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])
    