CONSTRUCTOR_PREDICATE_BOUNDARIES_INIT = 0
CONSTRUCTOR_BOUNDS_INIT = 1

LB_ID = 0
UB_ID = 1

PRED_VAL_ID = 0

V_ID = 0
C_ID = 1
D_ID = 2
PREDICATE_LB_ID = 3
PREDICATE_UB_ID = 4


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
                "test_inputs/fmnist_img/test_eval_input.mat"
            ]
    }