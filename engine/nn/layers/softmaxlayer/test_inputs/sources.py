SOFTMAXL_TEST_RSI_INIT = 0
SOFTMAXL_TEST_RMI_INIT = 1
SOFTMAXL_TEST_REACH_INIT = 2

SOFTMAXL_TEST_RSI_STAR_V_ID = 0
SOFTMAXL_TEST_RSI_STAR_C_ID = 1
SOFTMAXL_TEST_RSI_STAR_D_ID = 2
SOFTMAXL_TEST_RSI_STAR_PREDICATE_LB_ID = 3
SOFTMAXL_TEST_RSI_STAR_PREDICATE_UB_ID = 4

SOFTMAXL_TEST_RSI_IMGSTAR_V_ID = 5
SOFTMAXL_TEST_RSI_IMGSTAR_C_ID = 6
SOFTMAXL_TEST_RSI_IMGSTAR_D_ID = 7
SOFTMAXL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID = 8
SOFTMAXL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID = 9
SOFTMAXL_TEST_RSI_IMGSTAR_IM_LB_ID = 10
SOFTMAXL_TEST_RSI_IMGSTAR_IM_UB_ID = 11

SOFTMAXL_TEST_RMI_STAR_V_ID = 0
SOFTMAXL_TEST_RMI_STAR_C_ID = 1
SOFTMAXL_TEST_RMI_STAR_D_ID = 2
SOFTMAXL_TEST_RMI_STAR_PREDICATE_LB_ID = 3
SOFTMAXL_TEST_RMI_STAR_PREDICATE_UB_ID = 4

SOFTMAXL_TEST_RMI_IMGSTAR_V_ID = 5
SOFTMAXL_TEST_RMI_IMGSTAR_C_ID = 6
SOFTMAXL_TEST_RMI_IMGSTAR_D_ID = 7
SOFTMAXL_TEST_RMI_IMGSTAR_PREDICATE_LB_ID = 8
SOFTMAXL_TEST_RMI_IMGSTAR_PREDICATE_UB_ID = 9
SOFTMAXL_TEST_RMI_IMGSTAR_IM_LB_ID = 10
SOFTMAXL_TEST_RMI_IMGSTAR_IM_UB_ID = 11

SOFTMAXL_TEST_REACH_STAR_V_ID = 0
SOFTMAXL_TEST_REACH_STAR_C_ID = 1
SOFTMAXL_TEST_REACH_STAR_D_ID = 2
SOFTMAXL_TEST_REACH_STAR_PREDICATE_LB_ID = 3
SOFTMAXL_TEST_REACH_STAR_PREDICATE_UB_ID = 4

SOFTMAXL_TEST_REACH_IMGSTAR_V_ID = 5
SOFTMAXL_TEST_REACH_IMGSTAR_C_ID = 6
SOFTMAXL_TEST_REACH_IMGSTAR_D_ID = 7
SOFTMAXL_TEST_REACH_IMGSTAR_PREDICATE_LB_ID = 8
SOFTMAXL_TEST_REACH_IMGSTAR_PREDICATE_UB_ID = 9
SOFTMAXL_TEST_REACH_IMGSTAR_IM_LB_ID = 10
SOFTMAXL_TEST_REACH_IMGSTAR_IM_UB_ID = 11

sources = {
        SOFTMAXL_TEST_RSI_INIT : [
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SOFTMAXL_TEST_RMI_INIT : [
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SOFTMAXL_TEST_REACH_INIT : [
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/softmaxlayerlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
    }