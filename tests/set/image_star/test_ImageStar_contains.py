import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/imagestar/")
from imagestar import *

sys.path.insert(0, "../../../tests/test_utils/")
from utils import *

def process_exception(ex): 
    ex_type, ex_value, ex_traceback = sys.exc_info()
    trace_back = traceback.extract_tb(ex_traceback)
        
    stack_trace = ""
        
    for trace in trace_back:
        stack_trace = stack_trace + "File : %s ,\n Line : %d,\n Func.Name : %s,\n Message : %s\n" % (trace[0], trace[1], trace[2], trace[3])
                
    print("Exception type : %s " % ex_type.__name__)
    print("Exception message : %s" %ex_value)
    print("Stack trace : %s" %stack_trace)


class TestImageStarContains(unittest.TestCase):
    """
        Tests the 'contains' method
    """

    def test_contains_true(self):
        """
            Checks if the initialized ImageStar contains the given image
                       
            V -> Basis matrix
            C -> Predicate matrix
            d -> Predicate vector
            predicate_lb -> predicate lower bound
            predicate_ub -> predicate upper bound
            
            test_input -> the input image
        """
        
        test_V = np.reshape(read_csv_data(sources[CONTAINS_INIT][V_ID]), (28,28,1,785))
        test_C = np.reshape(read_csv_data(sources[CONTAINS_INIT][C_ID]), (1, 784))
        test_d = read_csv_data(sources[CONTAINS_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[CONTAINS_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[CONTAINS_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_input = read_csv_data(sources[CONTAINS_INIT][TRUE_INPUT_ID])
        
        test_result = None

        try:
            test_result = test_star.contains(test_input)
        except Exception as ex:
            completion_flag = False
            process_exception(ex)
        
        self.assertEqual(test_result, True)

    # def test_contains_fase(self):
    #     """
    #         Checks if the initialized ImageStar is empty
    #
    #         V -> Basis matrix
    #         C -> Predicate matrix
    #         d -> Predicate vector
    #         predicate_lb -> predicate lower bound
    #         predicate_ub -> predicate upper bound
    #     """
    #
    #     test_V = np.reshape(read_csv_data(sources[CONTAINS_INIT][V_ID]), (28,28,1,785))
    #     test_C = np.reshape(read_csv_data(sources[CONTAINS_INIT][C_ID]), (1, 784))
    #     test_d = read_csv_data(sources[CONTAINS_INIT][D_ID])
    #     test_predicate_lb = read_csv_data(sources[CONTAINS_INIT][PREDICATE_LB_ID])
    #     test_predicate_ub = read_csv_data(sources[CONTAINS_INIT][PREDICATE_UB_ID])
    #
    #     test_star = ImageStar(
    #             test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
    #         )
    #
    #     test_input = read_csv_data(sources[CONTAINS_INIT][FALSE_INPUT])
    #
    #     self.assertEqual(test_star.contains(test_input), False)

if __name__ == '__main__':
    unittest.main()
