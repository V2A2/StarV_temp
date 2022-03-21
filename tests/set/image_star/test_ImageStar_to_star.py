import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarToStar(unittest.TestCase):
    """
        Tests ImageStar constructor
    """

    def test_basic_to_star(self):
        """
            Tests the initialization with:
            lb -> lower bound
            ub -> upper bound
        """
        
        test_lb = read_csv_data(sources[TO_STAR_INIT][TEST_LB_ID])
        test_ub = read_csv_data(sources[TO_STAR_INIT][TEST_UB_ID])
        
        test_star = ImageStar(
                test_lb, test_ub
            )
    
        converted = test_star.to_star()

if __name__ == '__main__':
    unittest.main()
