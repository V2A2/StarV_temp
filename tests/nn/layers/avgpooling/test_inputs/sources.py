AVGP2D_TEST_CONSTRUCTOR_INIT = 0
AVGP2D_TEST_EVALUATE_INIT = 1
AVGP2D_TEST_RSI_INIT = 2
AVGP2D_TEST_RMI_INIT = 3
AVGP2D_TEST_REACH_INIT = 4

AVGP2D_TEST_POOL_SIZE_ID = 0
AVGP2D_TEST_PADDING_SIZE_ID = 1
AVGP2D_TEST_STRIDE_ID = 2
AVGP2D_TEST_NUM_INPUTS_ID = 3
AVGP2D_TEST_INPUT_NAMES_ID = 4
AVGP2D_TEST_NUM_OUTPUTS_ID = 5
AVGP2D_TEST_OUTPUT_NAMES_ID = 6

AVGP2D_TEST_RSI_V_ID = 7
AVGP2D_TEST_RSI_C_ID = 8
AVGP2D_TEST_RSI_D_ID = 9
AVGP2D_TEST_RSI_PREDICATE_LB_ID = 10
AVGP2D_TEST_RSI_PREDICATE_UB_ID = 11

AVGP2D_TEST_RMI_V_ID = 7
AVGP2D_TEST_RMI_C_ID = 8
AVGP2D_TEST_RMI_D_ID = 9
AVGP2D_TEST_RMI_PREDICATE_LB_ID = 10
AVGP2D_TEST_RMI_PREDICATE_UB_ID = 11

AVGP2D_TEST_REACH_V_ID = 7
AVGP2D_TEST_REACH_C_ID = 8
AVGP2D_TEST_REACH_D_ID = 9
AVGP2D_TEST_REACH_PREDICATE_LB_ID = 10
AVGP2D_TEST_REACH_PREDICATE_UB_ID = 11

sources = {
        AVGP2D_TEST_CONSTRUCTOR_INIT : [
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_pool_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_padding_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_stride.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_inputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_input_names.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_outputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_output_names.mat",
            ],
        AVGP2D_TEST_EVALUATE_INIT : [
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_pool_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_padding_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_stride.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_inputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_input_names.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_outputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_output_names.mat",
            ],
        AVGP2D_TEST_RSI_INIT : [
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_pool_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_padding_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_stride.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_inputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_input_names.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_outputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_output_names.mat",
                
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        AVGP2D_TEST_RMI_INIT : [
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_pool_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_padding_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_stride.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_inputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_input_names.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_outputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_output_names.mat",
                
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
        AVGP2D_TEST_REACH_INIT : [
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_pool_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_padding_size.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_stride.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_inputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_input_names.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_num_outputs.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_AvgPool2D_output_names.mat",
                
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/avgpooling/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
            ],
}