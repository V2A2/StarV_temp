BN_TEST_CONSTRUCTOR_INIT = 0
BN_TEST_EVALUATE_INIT = 1
BN_TEST_REACH_SINGLE_INPUT_INIT = 2
BN_TEST_REACH_MULTIPLE_INPUTS_INIT = 3
BN_TEST_REACH_INIT = 4

BN_TEST_TRAINED_VAR_ID = 0
BN_TEST_TRAINED_MEAN_ID = 1
BN_TEST_SCALE_ID = 2
BN_TEST_OFFSET_ID = 3
BN_TEST_EPSILON_ID = 4
BN_TEST_NUM_CHANNELS_ID = 5
BN_TEST_NUM_INPUTS_ID = 6
BN_TEST_NUM_OUTPUTS_ID = 7
BN_TEST_INPUT_NAMES_ID = 8
BN_TEST_OUTPUT_NAMES_ID = 9



BN_TEST_RSI_V_ID = 10
BN_TEST_RSI_C_ID = 11
BN_TEST_RSI_D_ID = 12
BN_TEST_RSI_PREDICATE_LB_ID = 13
BN_TEST_RSI_PREDICATE_UB_ID = 14



sources = {
        BN_TEST_CONSTRUCTOR_INIT : [
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_variance.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_mean.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_scale.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_offset.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_epsilon.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_channels.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_inputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_outputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_input_names.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_output_names.mat",
            ],
        BN_TEST_EVALUATE_INIT : [
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_variance.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_mean.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_scale.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_offset.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_epsilon.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_channels.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_inputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_outputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_input_names.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_output_names.mat"
            ],
        BN_TEST_REACH_SINGLE_INPUT_INIT : [
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_variance.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_mean.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_scale.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_offset.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_epsilon.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_channels.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_inputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_outputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_input_names.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_output_names.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_V.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_C.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_d.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_lb.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_ub.mat",
            ],
        BN_TEST_REACH_MULTIPLE_INPUTS_INIT : [
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_variance.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_mean.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_scale.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_offset.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_epsilon.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_channels.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_inputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_outputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_input_names.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_output_names.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_V.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_C.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_d.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_lb.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_ub.mat",
            ],
        BN_TEST_REACH_INIT : [
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_variance.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_trained_mean.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_scale.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_offset.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_epsilon.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_channels.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_inputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_num_outputs.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_input_names.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_BN_output_names.mat",
                
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_V.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_C.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_d.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_lb.mat",
                "tests/nn/layers/batchnormalization/test_inputs/test_star_mnist2/test_pred_ub.mat",
            ],
}