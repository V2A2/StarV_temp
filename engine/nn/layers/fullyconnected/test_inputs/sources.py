FC_REACH_SINGLE_INPUT_INIT = 0
FC_REACH_MULTIPLE_INPUTS_INIT = 1
FC_REACH_INIT = 2

V_ID = 0
C_ID = 1
D_ID = 2
PREDICATE_LB_ID = 3
PREDICATE_UB_ID = 4

FC_REACH_SINGLE_INPUT_WEIGHTS_ID = 5
FC_REACH_SINGLE_INPUT_BIAS_ID = 6

FC_REACH_MULTIPLE_INPUTS_WEIGHTS_ID = 5
FC_REACH_MULTIPLE_INPUTS_BIAS_ID = 6


FC_REACH_WEIGHTS_ID = 5
FC_REACH_BIAS_ID = 6

sources = {
        FC_REACH_SINGLE_INPUT_INIT : [
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_weights.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_bias.mat",
            ],
        FC_REACH_MULTIPLE_INPUTS_INIT : [
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_weights.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_bias.mat",
            ],
        FC_REACH_INIT : [
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_V.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_C.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_d.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_lb.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_star_mnist1/test_pred_ub.mat",
                
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_weights.mat",
                "engine/nn/layers/fullyconnected/test_inputs/test_FC_rsi_bias.mat",
            ],
}