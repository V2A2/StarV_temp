TANHL_TEST_RSI_INIT = 0
TANHL_TEST_RMI_INIT = 1
TANHL_TEST_REACH_INIT = 2

TANHL_TEST_RSI_STAR_V_ID = 0
TANHL_TEST_RSI_STAR_C_ID = 1
TANHL_TEST_RSI_STAR_D_ID = 2
TANHL_TEST_RSI_STAR_PREDICATE_LB_ID = 3
TANHL_TEST_RSI_STAR_PREDICATE_UB_ID = 4

TANHL_TEST_RSI_IMGSTAR_V_ID = 5
TANHL_TEST_RSI_IMGSTAR_C_ID = 6
TANHL_TEST_RSI_IMGSTAR_D_ID = 7
TANHL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID = 8
TANHL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID = 9
TANHL_TEST_RSI_IMGSTAR_IM_LB_ID = 10
TANHL_TEST_RSI_IMGSTAR_IM_UB_ID = 11

TANHL_TEST_RMI_STAR_V_ID = 0
TANHL_TEST_RMI_STAR_C_ID = 1
TANHL_TEST_RMI_STAR_D_ID = 2
TANHL_TEST_RMI_STAR_PREDICATE_LB_ID = 3
TANHL_TEST_RMI_STAR_PREDICATE_UB_ID = 4

TANHL_TEST_RMI_IMGSTAR_V_ID = 5
TANHL_TEST_RMI_IMGSTAR_C_ID = 6
TANHL_TEST_RMI_IMGSTAR_D_ID = 7
TANHL_TEST_RMI_IMGSTAR_PREDICATE_LB_ID = 8
TANHL_TEST_RMI_IMGSTAR_PREDICATE_UB_ID = 9
TANHL_TEST_RMI_IMGSTAR_IM_LB_ID = 10
TANHL_TEST_RMI_IMGSTAR_IM_UB_ID = 11

TANHL_TEST_REACH_STAR_V_ID = 0
TANHL_TEST_REACH_STAR_C_ID = 1
TANHL_TEST_REACH_STAR_D_ID = 2
TANHL_TEST_REACH_STAR_PREDICATE_LB_ID = 3
TANHL_TEST_REACH_STAR_PREDICATE_UB_ID = 4

TANHL_TEST_REACH_IMGSTAR_V_ID = 5
TANHL_TEST_REACH_IMGSTAR_C_ID = 6
TANHL_TEST_REACH_IMGSTAR_D_ID = 7
TANHL_TEST_REACH_IMGSTAR_PREDICATE_LB_ID = 8
TANHL_TEST_REACH_IMGSTAR_PREDICATE_UB_ID = 9
TANHL_TEST_REACH_IMGSTAR_IM_LB_ID = 10
TANHL_TEST_REACH_IMGSTAR_IM_UB_ID = 11

sources = {
        TANHL_TEST_RSI_INIT : [
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        TANHL_TEST_RMI_INIT : [
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        TANHL_TEST_REACH_INIT : [
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/relulayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
    }