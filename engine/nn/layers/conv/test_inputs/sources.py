CONV2D_TEST_CONSTRUCTOR_INIT = 0
CONV2D_TEST_EVAL_INIT = 1
CONV2D_TEST_RSI_INIT = 2
CONV2D_TEST_RMI_INIT = 3
CONV2D_TEST_REACH_INIT = 4

CONV2D_TEST_WEIGHTS_ID = 0
CONV2D_TEST_BIASS_ID = 1
CONV2D_TEST_FILTER_SIZE_ID = 2
CONV2D_TEST_PADDING_SIZE_ID = 3
CONV2D_TEST_PADDING_MODE_ID = 4
CONV2D_TEST_DILATION_FACTOR_ID = 5
CONV2D_TEST_STRIDE_ID = 6
CONV2D_TEST_NUM_FILTERS_ID = 7
CONV2D_TEST_NUM_CHANNELS_ID = 8
CONV2D_TEST_NUM_INPUTS_ID = 9
CONV2D_TEST_INPUT_NAMES_ID = 10
CONV2D_TEST_NUM_OUTPUTS_ID = 11
CONV2D_TEST_OUTPUT_NAMES_ID = 12

CONV2D_TEST_RSI_V_ID = 13
CONV2D_TEST_RSI_C_ID = 14
CONV2D_TEST_RSI_D_ID = 15
CONV2D_TEST_RSI_PREDICATE_LB_ID = 16
CONV2D_TEST_RSI_PREDICATE_UB_ID = 17
CONV2D_TEST_RSI_IM_LB_ID = 18
CONV2D_TEST_RSI_IM_UB_ID = 19

CONV2D_TEST_RMI_V_ID = 13
CONV2D_TEST_RMI_C_ID = 14
CONV2D_TEST_RMI_D_ID = 15
CONV2D_TEST_RMI_PREDICATE_LB_ID = 16
CONV2D_TEST_RMI_PREDICATE_UB_ID = 17
CONV2D_TEST_RMI_IM_LB_ID = 18
CONV2D_TEST_RMI_IM_UB_ID = 19

CONV2D_TEST_REACH_V_ID = 13
CONV2D_TEST_REACH_C_ID = 14
CONV2D_TEST_REACH_D_ID = 15
CONV2D_TEST_REACH_PREDICATE_LB_ID = 16
CONV2D_TEST_REACH_PREDICATE_UB_ID = 17
CONV2D_TEST_REACH_IM_LB_ID = 18
CONV2D_TEST_REACH_IM_UB_ID = 19

sources = {
        CONV2D_TEST_CONSTRUCTOR_INIT : [
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_weights.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_bias.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_filter_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_mode.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_dilation_factor.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_stride.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_filters.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_channels.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_inputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_input_names.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_outputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_output_names.mat"
            ],
        CONV2D_TEST_EVAL_INIT : [
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_weights.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_bias.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_filter_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_mode.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_dilation_factor.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_stride.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_filters.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_channels.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_inputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_input_names.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_outputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_output_names.mat"
            ],
        CONV2D_TEST_RSI_INIT : [
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_weights.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_bias.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_filter_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_mode.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_dilation_factor.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_stride.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_filters.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_channels.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_inputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_input_names.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_outputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_output_names.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        CONV2D_TEST_RMI_INIT : [
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_weights.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_bias.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_filter_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_mode.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_dilation_factor.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_stride.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_filters.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_channels.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_inputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_input_names.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_outputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_output_names.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
        CONV2D_TEST_REACH_INIT : [
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_weights.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_bias.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_filter_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_size.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_padding_mode.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_dilation_factor.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_stride.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_filters.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_channels.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_inputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_input_names.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_num_outputs.mat",
                "engine/nn/layers/conv2d/test_inputs/test_Conv2D_output_names.mat",
                
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_V.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_C.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_d.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_pred_ub.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_lb.mat",
                "engine/nn/layers/conv2d/test_inputs/test_imagestar_fmnist1/test_im_ub.mat",
            ],
}