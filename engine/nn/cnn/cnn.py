CNN_ATTRIBUTES_NUM = 5

CNN_NAME = 0
CNN_LAYERS = 1
CNN_INPUT_SIZE = 2
CNN_OUTPUT_SIZE = 3
CNN_OUTPUT_SET = 4

CNN_AGS_NAME = 0
CNN_AGS_LAYERS = 1
CNN_AGS_INPUT_SIZE = 2
CNN_AGS_OUTPUT_SIZE = 3

CNN_EVAL_ARGS_INPUT = 0

CNN_REACH_ARGS_INPUT = 0

CNN_CLASSIFY_ARGS_INPUT = 0
CNN_CLASSIFY_ARGS_METHOD = 1

CNN_VERIFY_ROBUSTNESS_ARGS_INPUT = 0
CNN_VERIFY_ROBUSTNESS_ARGS_METHOD = 1

CNN_EVALUATE_ROBUSTNESS_ARGS_INPUT = 0
CNN_EVALUATE_ROBUSTNESS_ARGS_METHOD = 1

CNN_IS_ROBUST_ARGS_OUTPUT_SET = 0
CNN_IS_ROBUST_ARGS_CORRECT_ID = 1

CNN_CHECK_ROBUST_ARGS_OUTPUT_SET = 0

CNN_CLASSIFY_OUTPUT_SET_ARGS_OUTPUT_SET = 0

CNN_ERRMSG_INVALID_NUMBER_OF_INPUTS = 'Invalid number of inputs, should be 0, 3, or 4'
CNN_DEFAULT_REACH_METHOD = 'approx-star'

class CNN:
    
    def __init__(self, *args):
        """
            Constructor
        """
        
        self.attributes = []
        
        for i in range(CNN_ATTRIBUTES_NUM):
            self.attributes.append(np.array([]))
            
        if len(args) == CNN_FULL_ARGS_LEN:
            self.attributes[CNN_NAME] = args[CNN_ARS_NAME]
            self.attributes[CNN_LAYERS] = args[CNN_ARS_LAYERS]
            self.attritubets[CNN_INPUT_SIZE] = self.attributes[CNN_ARGS_LAYERS][0].get_input_size()
            self.attritubets[CNN_OUTPUT_SIZE] = self.attributes[CNN_ARGS_LAYERS].get_last().get_output_size()    
        elif len(args) == CNN_CALC_ARGS_LEN:
            args = self.offset_args(args, CNN_CALC_ARGS_OFFSET)
            self.attributes[CNN_NAME] = CNN_DEFAULT_NAME
            self.atributes[CNN_LAYERS] = args[CNN_ARGS_LAYERS]
            self.attritubets[CNN_INPUT_SIZE] = self.attributes[CNN_ARGS_LAYERS][0].get_input_size()
            self.attritubets[CNN_OUTPUT_SIZE] = self.attributes[CNN_ARGS_LAYERS].get_last().get_output_size()         
        elif len(args) == CNN_EMPTY_ARGS_LEN:
            self.attributes[CNN_NAME] = CNN_DEFAULT_NAME
            self.atributes[CNN_LAYERS] = CNN_DEFAULT_LAYERS
            self.attritubets[CNN_INPUT_SIZE] = CNN_DEFAULT_INPUT_SIZE
            self.attritubets[CNN_OUTPUT_SIZE] = CNN_DEFAULT_OUTPUT_SIZE         
        else:
            raise Exception(CNN_ERRMSG_INVALID_NUMBER_OF_INPUTS)
        
        def evaluate(self, *args):
            """
                Evaluates the CNN
                
                x : np.array([*np.array([*double])]) -> input
                
                returns the output obtained after the final layer
            """

            y = args[CNN_EVAL_ARGS_INPUT]

            for i in range(len(self.attributes[CNN_LAYERS])):
                y = self.attributes[CNN_LAYERS][i].evaluate(y)
                
        def reach(self, *args):
            """
                Performs reachability analysis on the input
                input : (Image)Star or (Image)Zono -> input
                method : string -> ...
                options : *not implemented*
                
                returns the output set and reachability analysis time
            """

            rs = []
            reach_time = 0
            
            # todo: deep copy?
            y = args[CNN_REACH_ARGS_INPUT]
            
            for i in range(self.attributes[CNN_LAYERS]):
                y = self.attributes[CNN_LAYERS][i].reach(y)
                rs.append(y)
                
            return rs, reach_time
            
        def classify(self, *args):
            """
                Classifies the input image
                
                input : ImageStar or ImageZono -> input
                options : *not implemented*
                
                returns the output index of classified object
            """
            
            input = args[CNN_CLASSIFY_ARGS_INPUT]
            method = (CNN_DEFAULT_REACH_METHOD) if len(args) == 1 else args[CNN_CLASSIFY_ARGS_METHOD] 
            
            if not isinstance(input, ImageStar) or not isinstance(input, ImageZono):
                y = self.evaluate(input)
                y = np.reshape(y, (input.get_output_size(), 1))
                [_, label_id] = max(y)
            else:
                output_sets = self.reach(input, method)
                rs_num = len(output_sets) if type(output_sets) == list else 1
                
                label_id = [np.array([]) for i in range(rs_num)]
                
                for i in range(rs_num):
                    current_rs = ImageStar.reshape(res, np.array([self.attributes[CNN_OUTPUT_SIZE][0], 1, 1]))
                    max_id = current_rs.get_localMax_indes(np.array([1, 1], np.array([self.attributes[CNN_OUTPUT_SIZE][0], 1, 1])))
                    label_id[i] = max_id[i]
                    
            return label_id
                
        def verify_robustness(self, *args):
            """
                Verifies the robustness of the network
                
                input : ImageStar or ImageZono -> input
                correct_id : np.array([*int]) -> correct class
                method : string -> reachability method
                
                returns if the network is : 1 -> robust,
                                            0 -> is not robust,
                                            2 -> unknown,
                                            and a set of counterexamples
            """
            
            input = args[CNN_VERIFY_ROBUSTNESS_ARGS_INPUT]
            correct_id = [CNN_VERIFY_ROBUSTNESSY_ARGS_CORRECT_ID]
            method = (CNN_DEFAULT_REACH_METHOD) if len(args) == 1 else args[CNN_VERIFY_ROBUSTNESS_ARGS_METHOD]
            
            label_id = self.classify(input, method)
            n = len(label_id)
            
            counter_ex = []
            robust = -1
            
            incorrect_id_list = []
            
            for i in range(n):
                ids = label_id[i]
                incorrect_ids = np.argwhere(ids != correct_id)
                
                incorrect_id_list = np.append(incorrect_id_list, incorrect_ids)
                
                if not self.isempty(incorrect_ids):
                    if not isinstance(input, ImageStar):
                        counter_ex = input
                    elif method == 'exact-star':
                        current_reach_set = self.attributes[CNN_OUTPUT_SET]
                        for j in range(len(incorrect_ids)):
                            [new_C, new_d] = ImageStar.add_constraint(current_reach_set, np.array[(1, 1, correct_id)], np.array([1, 1, incorrect_ids[j]]))
                            counter_IS = ImageStar(input.get_V(), new_C, new_D, input.get_pred_lb(), input.get_pred_ub())
                            counter_ex.append(counter_IS)
            
            if self.isempty(incorrect_id_list):
                robust = 1
            else:
                if method == 'exact-star':
                    robust = 0
                else:
                    robust = 2
                    
            return robust, counter_ex
        
    def evaluate_robustness(self, *args):
        """
            Evaluates robustness of the network on a set of inputs
            
            input : [*ImageStar] -> a set of inputs
            correct_ids : np.array([*np.array([])) -> a set of correct labels for the given inputs
            method : string
            options : *not implemented*
            
            returns a robustness value (in percentages)
        """
        
        inputs = args[CNN_EVALUATE_ROBUSTNESS_ARGS_INPUT]
        correct_id = [CNN_EVALUATE_ROBUSTNES_ARGS_CORRECT_IDS]
        method = (CNN_DEFAULT_REACH_METHOD) if len(args) == 1 else args[CNN_EVALUATE_ROBUSTNES_ARGS_ARGS_METHOD]
        
        count = np.zeros(len(inputs))

        if method != 'exact-star':
            output_sets = self.reach(inputs, method)
            
            for i in range(len(inputs)):
                count[i] = CNN.is_robust(output_sets[i], correct_ids[i])
        elif method == 'exact-star':
            
            for i in range(len(inputs)):
                output_sets = self.reach(inputs, method)
                
                current_num = 0
                for j in range(len(output_sets)):
                    current_num += CNN.is_robust(output_sets[j], correct_id[i])
                    
        return np.sum(count) / len(inputs)
    
    def is_robust(self, *args):
        """
            Check robustness using the output set
            
            output_set : ImageStar -> the output set that will be checked
            correct_id : int -> correct id of the classified output
            
            returns whether the network is robust
        """
                    
        counter = 0
        
        output_set = args[CNN_IS_ROBUST_ARGS_OUTPUT_SET]
        correct_id = args[CNN_IS_ROBUST_ARGS_CORRECT_ID]
        
        for i in range(self.attributes[CNN_OUTPUT_SET].get_num_channel()):
            if correct_id != i:
                if output_set.is_p1_larger_p2(np.array([1, 1, i]), np.array([1, 1, correct_id])):
                    return 0
                else:
                    counter += 1
        
        if counter == output_set.get_num_channel() - 1:
            return 1
                    
    def check_robust(self, *args):
        """
            Check robustness using the output set
            
            output_set : ImageStar -> the output set that will be checked
            correct_id : int -> correct id of the classified output
            
            returns if the network is : 1 -> robust,
                                        0 -> is not robust,
                                        2 -> unknown,
                                        and a set of possible candidates
        """
        
        reachable_set = args[CNN_CHECK_ROBUST_ARGS_OUTPUT_SET].to_star()
        
        [lb, ub] = reachable_set.estimate_ranges()
        [_, max_ub_id] = max(ub)
        
        candidates = []
        is_robust = -1
        
        if max_ub_id != correct_id:
            is_robust = 2
            candidates = max_ub_id
        else:
            max_val = lb[correct_id]
            max_cd = np.argwhere(ub > max_val)
            max_cd[max_cd == correct_id] = []
            
            if self.isempty(max_cd):
                is_robust = 1
            else:
                n = len(max_cd)
                C1 = reachable_set.get_V()[max_cd, 1: reachable_set.get_nVar() + 1] - np.multiply(np.ones((n, 1)), reachable_set.get_V()[correct_id, 1:reachable_set.get_nVar() + 1])
                d1 = -reachable_set.get_V()[max_cd, 0] + np.multiply(np.ones((n, 1)), reachable_set.get_V()[correct_id, 0])
                
                S = Star(reachable_set.get_V(), np.vstack((reachable_set.get_C(), C1)), np.vstack((reachable_set.get_d(), d1)), reachable_set.get_pred_lb(), reachable_set.get_pred_ub())
                
                if S.isEmptySet():
                    is_robust = 2
                    candidates = max_cd
                else:
                    count = 0
                    
                    for i in range(n):
                        if reachable_set.is_p1_larger_p2(max_cd[i], correct_id):
                            is_robust = 2
                            candidates = max_cd[i]
                            break
                        else:
                            count += 1
                    if count == n:
                        is_robust = 1
                        
        return is_robust, candidates
            
    def classify_output_set(self, *args):
        """
            Classifies the output set
            
            output_set : ImageStar or ImageZono -> the output reachable set
            
            returns the classified id of the given output, if the given output set 
                    cannot be classified then the id is chosen so that it would
                    correspond to the output that has the maximum value
        """
        
        output_set = args[CNN_CLASSIFY_OUTPUT_SET_ARGS_OUTPUT_SET]
        classified_id = None
        
        [lb, ub] = output_set.estimate_ranges()
        [max_lb_id, max_lb] = np.max(lb)
        n = lb.shape[0]
        
        ub1 = ub
        ub1[max_lb_id] = []
        ub1 = ub1 > max_lb
        
        if sum(ub1) == 0:
            classified_id = max_lb_id
        else:
            classified_id = max_lb_id
            [_, act_lb] = output_set.get_range(0, 0, max_lb_id)
            
            for i in range(n):
                if ub[i] > max_lb and i != max_lb_id:
                    [ub_id, _] = output_set.get_range(0, 0, i)
                    if ub_id > act_lb:
                        classified_id = np.append(classified_id, i)
                        
        return classified_id
            
            
        
        
        
            