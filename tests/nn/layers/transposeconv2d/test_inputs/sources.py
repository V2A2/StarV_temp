CONVTR2D_TEST_CONSTRUCTOR_INIT = 0
CONVTR2D_TEST_EVAL_INIT = 1
CONVTR2D_TEST_RSI_INIT = 2
CONVTR2D_TEST_RMI_INIT = 3
CONVTR2D_TEST_REACH_INIT = 4

CONVTR2D_TEST_WEIGHTS_ID = 0
CONVTR2D_TEST_BIASS_ID = 1
CONVTR2D_TEST_FILTER_SIZE_ID = 2
CONVTR2D_TEST_PADDING_SIZE_ID = 3
CONVTR2D_TEST_PADDING_MODE_ID = 4
CONVTR2D_TEST_DILATION_FACTOR_ID = 5
CONVTR2D_TEST_STRIDE_ID = 6
CONVTR2D_TEST_NUM_FILTERS_ID = 7
CONVTR2D_TEST_NUM_CHANNELS_ID = 8
CONVTR2D_TEST_NUM_INPUTS_ID = 9
CONVTR2D_TEST_INPUT_NAMES_ID = 10
CONVTR2D_TEST_NUM_OUTPUTS_ID = 11
CONVTR2D_TEST_OUTPUT_NAMES_ID = 12

CONVTR2D_TEST_RSI_V_ID = 13
CONVTR2D_TEST_RSI_C_ID = 14
CONVTR2D_TEST_RSI_D_ID = 15
CONVTR2D_TEST_RSI_PREDICATE_LB_ID = 16
CONVTR2D_TEST_RSI_PREDICATE_UB_ID = 17
CONVTR2D_TEST_RSI_IM_LB_ID = 18
CONVTR2D_TEST_RSI_IM_UB_ID = 19

CONVTR2D_TEST_RMI_V_ID = 13
CONVTR2D_TEST_RMI_C_ID = 14
CONVTR2D_TEST_RMI_D_ID = 15
CONVTR2D_TEST_RMI_PREDICATE_LB_ID = 16
CONVTR2D_TEST_RMI_PREDICATE_UB_ID = 17
CONVTR2D_TEST_RMI_IM_LB_ID = 18
CONVTR2D_TEST_RMI_IM_UB_ID = 19

CONVTR2D_TEST_REACH_V_ID = 13
CONVTR2D_TEST_REACH_C_ID = 14
CONVTR2D_TEST_REACH_D_ID = 15
CONVTR2D_TEST_REACH_PREDICATE_LB_ID = 16
CONVTR2D_TEST_REACH_PREDICATE_UB_ID = 17
CONVTR2D_TEST_REACH_IM_LB_ID = 18
CONVTR2D_TEST_REACH_IM_UB_ID = 19

sources = {
        CONVTR2D_TEST_CONSTRUCTOR_INIT : [
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_weights.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_bias.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_filter_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_mode.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_dilation_factor.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_stride.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_filters.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_channels.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_inputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_input_names.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_outputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_output_names.mat"
            ],
        CONVTR2D_TEST_EVAL_INIT : [
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_weights.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_bias.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_filter_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_mode.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_dilation_factor.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_stride.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_filters.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_channels.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_inputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_input_names.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_outputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_output_names.mat"
            ],
        CONVTR2D_TEST_RSI_INIT : [
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_weights.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_bias.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_filter_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_mode.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_dilation_factor.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_stride.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_filters.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_channels.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_inputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_input_names.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_outputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_output_names.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        CONVTR2D_TEST_RMI_INIT : [
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_weights.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_bias.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_filter_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_mode.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_dilation_factor.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_stride.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_filters.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_channels.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_inputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_input_names.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_outputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_output_names.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        CONVTR2D_TEST_REACH_INIT : [
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_weights.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_bias.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_filter_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_size.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_padding_mode.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_dilation_factor.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_stride.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_filters.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_channels.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_inputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_input_names.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_num_outputs.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_ConvTr2D_output_names.mat",
                
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "tests/nn/layers/transposeconv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
}