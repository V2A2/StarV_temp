import unittest

from test_inputs.sources import *

class TestImageStarToStar(unittest.TestCase):
    """
        Tests the conversion from ImageStar to Star
    """

    def test_basic_to_star(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """
        
        test_lb = read_csv_data(sources[TO_STAR_INIT][TEST_LB_ID])
        test_ub = read_csv_data(sources[TO_STAR_INIT][TEST_UB_ID])
    
        completion_flag = True
    
        try:
            test_star = ImageStar(
                    test_lb, test_ub
                )
            
            converted = test_star.to_star()
        except Exception as ex:
            completion_flag = False
            process_exception(ex)

            
        self.assertEqual(completion_flag, True)

if __name__ == '__main__':
    unittest.main()
