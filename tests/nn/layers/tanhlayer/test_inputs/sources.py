SIGMOIDL_TEST_RSI_INIT = 0
SIGMOIDL_TEST_RMI_INIT = 1
SIGMOIDL_TEST_REACH_INIT = 2

SIGMOIDL_TEST_RSI_STAR_V_ID = 0
SIGMOIDL_TEST_RSI_STAR_C_ID = 1
SIGMOIDL_TEST_RSI_STAR_D_ID = 2
SIGMOIDL_TEST_RSI_STAR_PREDICATE_LB_ID = 3
SIGMOIDL_TEST_RSI_STAR_PREDICATE_UB_ID = 4

SIGMOIDL_TEST_RSI_IMGSTAR_V_ID = 5
SIGMOIDL_TEST_RSI_IMGSTAR_C_ID = 6
SIGMOIDL_TEST_RSI_IMGSTAR_D_ID = 7
SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID = 8
SIGMOIDL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID = 9
SIGMOIDL_TEST_RSI_IMGSTAR_IM_LB_ID = 10
SIGMOIDL_TEST_RSI_IMGSTAR_IM_UB_ID = 11

SIGMOIDL_TEST_RMI_STAR_V_ID = 0
SIGMOIDL_TEST_RMI_STAR_C_ID = 1
SIGMOIDL_TEST_RMI_STAR_D_ID = 2
SIGMOIDL_TEST_RMI_STAR_PREDICATE_LB_ID = 3
SIGMOIDL_TEST_RMI_STAR_PREDICATE_UB_ID = 4

SIGMOIDL_TEST_RMI_IMGSTAR_V_ID = 5
SIGMOIDL_TEST_RMI_IMGSTAR_C_ID = 6
SIGMOIDL_TEST_RMI_IMGSTAR_D_ID = 7
SIGMOIDL_TEST_RMI_IMGSTAR_PREDICATE_LB_ID = 8
SIGMOIDL_TEST_RMI_IMGSTAR_PREDICATE_UB_ID = 9
SIGMOIDL_TEST_RMI_IMGSTAR_IM_LB_ID = 10
SIGMOIDL_TEST_RMI_IMGSTAR_IM_UB_ID = 11

SIGMOIDL_TEST_REACH_STAR_V_ID = 0
SIGMOIDL_TEST_REACH_STAR_C_ID = 1
SIGMOIDL_TEST_REACH_STAR_D_ID = 2
SIGMOIDL_TEST_REACH_STAR_PREDICATE_LB_ID = 3
SIGMOIDL_TEST_REACH_STAR_PREDICATE_UB_ID = 4

SIGMOIDL_TEST_REACH_IMGSTAR_V_ID = 5
SIGMOIDL_TEST_REACH_IMGSTAR_C_ID = 6
SIGMOIDL_TEST_REACH_IMGSTAR_D_ID = 7
SIGMOIDL_TEST_REACH_IMGSTAR_PREDICATE_LB_ID = 8
SIGMOIDL_TEST_REACH_IMGSTAR_PREDICATE_UB_ID = 9
SIGMOIDL_TEST_REACH_IMGSTAR_IM_LB_ID = 10
SIGMOIDL_TEST_REACH_IMGSTAR_IM_UB_ID = 11

sources = {
        SIGMOIDL_TEST_RSI_INIT : [
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SIGMOIDL_TEST_RMI_INIT : [
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SIGMOIDL_TEST_REACH_INIT : [
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
    }