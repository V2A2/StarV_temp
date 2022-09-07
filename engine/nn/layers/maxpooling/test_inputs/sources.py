MAXP2D_TEST_CONSTRUCTOR_INIT = 0
MAXP2D_TEST_EVALUATE_INIT = 1
MAXP2D_TEST_GET_START_POINTS_INIT = 2
MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT = 3
MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT = 4
MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT = 5
MAXP2D_TEST_RSI_INIT = 6
MAXP2D_TEST_RSI_APPROX_INIT = 7
MAXP2D_TEST_RMI_INIT = 8
MAXP2D_TEST_RMI_APPROX_INIT = 9
MAXP2D_TEST_RSI_ZONO_INIT = 10
MAXP2D_TEST_RMI_ZONO_INIT = 11
MAXP2D_TEST_REACH_INIT = 12

MAXP2D_TEST_POOL_SIZE_ID = 0
MAXP2D_TEST_PADDING_SIZE_ID = 1
MAXP2D_TEST_STRIDE_ID = 2
MAXP2D_TEST_NUM_INPUTS_ID = 3
MAXP2D_TEST_INPUT_NAMES_ID = 4
MAXP2D_TEST_NUM_OUTPUTS_ID = 5
MAXP2D_TEST_OUTPUT_NAMES_ID = 6

MAXP2D_TEST_GET_START_POINTS_V_ID = 7
MAXP2D_TEST_GET_ZERO_PADDING_INPUT_V_ID = 7
MAXP2D_TEST_GET_SIZE_MAX_MAP_V_ID = 7

MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_V_ID = 7
MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_C_ID = 8
MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_D_ID = 9
MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_LB_ID = 10
MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_PREDICATE_UB_ID = 11

MAXP2D_TEST_RSI_V_ID = 7
MAXP2D_TEST_RSI_C_ID = 8
MAXP2D_TEST_RSI_D_ID = 9
MAXP2D_TEST_RSI_PREDICATE_LB_ID = 10
MAXP2D_TEST_RSI_PREDICATE_UB_ID = 11

MAXP2D_TEST_RSI_APPROX_V_ID = 7
MAXP2D_TEST_RSI_APPROX_C_ID = 8
MAXP2D_TEST_RSI_APPROX_D_ID = 9
MAXP2D_TEST_RSI_APPROX_PREDICATE_LB_ID = 10
MAXP2D_TEST_RSI_APPROX_PREDICATE_UB_ID = 11

MAXP2D_TEST_RMI_V_ID = 7
MAXP2D_TEST_RMI_C_ID = 8
MAXP2D_TEST_RMI_D_ID = 9
MAXP2D_TEST_RMI_PREDICATE_LB_ID = 10
MAXP2D_TEST_RMI_PREDICATE_UB_ID = 11

MAXP2D_TEST_RMI_APPROX_V_ID = 7
MAXP2D_TEST_RMI_APPROX_C_ID = 8
MAXP2D_TEST_RMI_APPROX_D_ID = 9
MAXP2D_TEST_RMI_APPROX_PREDICATE_LB_ID = 10
MAXP2D_TEST_RMI_APPROX_PREDICATE_UB_ID = 11

MAXP2D_TEST_RSI_ZONO_V_ID = 7
MAXP2D_TEST_RSI_ZONO_PREDICATE_LB_ID = 8
MAXP2D_TEST_RSI_ZONO_PREDICATE_UB_ID = 9

MAXP2D_TEST_RMI_ZONO_V_ID = 7
MAXP2D_TEST_RMI_ZONO_PREDICATE_LB_ID = 8
MAXP2D_TEST_RMI_ZONO_PREDICATE_UB_ID = 9

MAXP2D_TEST_REACH_V_ID = 7
MAXP2D_TEST_REACH_C_ID = 8
MAXP2D_TEST_REACH_D_ID = 9
MAXP2D_TEST_REACH_PREDICATE_LB_ID = 10
MAXP2D_TEST_REACH_PREDICATE_UB_ID = 11

sources = {
        MAXP2D_TEST_CONSTRUCTOR_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
            ],
        MAXP2D_TEST_EVALUATE_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
            ],
        MAXP2D_TEST_GET_START_POINTS_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
            ],
        MAXP2D_TEST_GET_ZERO_PADDING_INPUT_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
            ],
        MAXP2D_TEST_GET_SIZE_MAX_MAP_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
            ],
        MAXP2D_TEST_GET_ZERO_PADDING_IMGSTAR_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RSI_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RSI_APPROX_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RMI_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RMI_APPROX_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RSI_ZONO_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_RMI_ZONO_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        MAXP2D_TEST_REACH_INIT : [
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_pool_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_padding_size.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_stride.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_inputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_input_names.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_num_outputs.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_MaxPool2D_output_names.mat",
                
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/maxpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
}