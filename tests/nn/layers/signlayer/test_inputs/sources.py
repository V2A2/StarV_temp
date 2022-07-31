SIGNL_TEST_RSI_INIT = 0
SIGNL_TEST_RMI_INIT = 1
SIGNL_TEST_REACH_INIT = 2



SIGNL_TEST_RSI_STAR_V_ID = 0
SIGNL_TEST_RSI_STAR_C_ID = 1
SIGNL_TEST_RSI_STAR_D_ID = 2
SIGNL_TEST_RSI_STAR_PREDICATE_LB_ID = 3
SIGNL_TEST_RSI_STAR_PREDICATE_UB_ID = 4
SIGNL_TEST_RSI_STAR_ZONO_V_ID = 5
SIGNL_TEST_RSI_STAR_ZONO_C_ID = 6

SIGNL_TEST_RSI_IMGSTAR_V_ID = 7
SIGNL_TEST_RSI_IMGSTAR_C_ID = 8
SIGNL_TEST_RSI_IMGSTAR_D_ID = 9
SIGNL_TEST_RSI_IMGSTAR_PREDICATE_LB_ID = 10
SIGNL_TEST_RSI_IMGSTAR_PREDICATE_UB_ID = 11
SIGNL_TEST_RSI_IMGSTAR_IM_LB_ID = 12
SIGNL_TEST_RSI_IMGSTAR_IM_UB_ID = 13



SIGNL_TEST_RMI_STAR_V_ID = 0
SIGNL_TEST_RMI_STAR_C_ID = 1
SIGNL_TEST_RMI_STAR_D_ID = 2
SIGNL_TEST_RMI_STAR_PREDICATE_LB_ID = 3
SIGNL_TEST_RMI_STAR_PREDICATE_UB_ID = 4
SIGNL_TEST_RMI_STAR_ZONO_V_ID = 5
SIGNL_TEST_RMI_STAR_ZONO_C_ID = 6

SIGNL_TEST_RMI_IMGSTAR_V_ID = 7
SIGNL_TEST_RMI_IMGSTAR_C_ID = 8
SIGNL_TEST_RMI_IMGSTAR_D_ID = 9
SIGNL_TEST_RMI_IMGSTAR_PREDICATE_LB_ID = 10
SIGNL_TEST_RMI_IMGSTAR_PREDICATE_UB_ID = 11
SIGNL_TEST_RMI_IMGSTAR_IM_LB_ID = 12
SIGNL_TEST_RMI_IMGSTAR_IM_UB_ID = 13



SIGNL_TEST_REACH_STAR_V_ID = 0
SIGNL_TEST_REACH_STAR_C_ID = 1
SIGNL_TEST_REACH_STAR_D_ID = 2
SIGNL_TEST_REACH_STAR_PREDICATE_LB_ID = 3
SIGNL_TEST_REACH_STAR_PREDICATE_UB_ID = 4
SIGNL_TEST_REACH_STAR_ZONO_V_ID = 5
SIGNL_TEST_REACH_STAR_ZONO_C_ID = 6

SIGNL_TEST_REACH_IMGSTAR_V_ID = 7
SIGNL_TEST_REACH_IMGSTAR_C_ID = 8
SIGNL_TEST_REACH_IMGSTAR_D_ID = 9
SIGNL_TEST_REACH_IMGSTAR_PREDICATE_LB_ID = 10
SIGNL_TEST_REACH_IMGSTAR_PREDICATE_UB_ID = 11
SIGNL_TEST_REACH_IMGSTAR_IM_LB_ID = 12
SIGNL_TEST_REACH_IMGSTAR_IM_UB_ID = 13



sources = {
        SIGNL_TEST_RSI_INIT : [
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_c.mat",
                
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SIGNL_TEST_RMI_INIT : [
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_c.mat",
                
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        SIGNL_TEST_REACH_INIT : [
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_star_mnist1/test_Z_c.mat",
                
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/signlayer/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
    }