import unittest

import sys

from test_inputs.sources import *

sys.path.insert(0, "../../../engine/set/")

from imagestar import *

class TestImageStarAffineMap(unittest.TestCase):
    """
        Tests ImageStar constructor
    """

    def test_affine_map(self):
        """
            scale : float -> affine map scale
            offset : np.array([*float]) -> affine map offset
            
            return -> ImageStar .* scale + offset
        """
                
        test_am_output = read_csv_data(sources[AFFINEMAP_INIT][AFFINEMAP_OUTPUT_ID])
                
        test_V = np.reshape(read_csv_data(sources[AFFINEMAP_INIT][V_ID]), (1, 1, 72, 1007))
        test_C = read_csv_data(sources[AFFINEMAP_INIT][C_ID])
        test_d = read_csv_data(sources[AFFINEMAP_INIT][D_ID])
        test_predicate_lb = read_csv_data(sources[AFFINEMAP_INIT][PREDICATE_LB_ID])
        test_predicate_ub = read_csv_data(sources[AFFINEMAP_INIT][PREDICATE_UB_ID])
        
        test_star = ImageStar(
                test_V, test_C, test_d, test_predicate_lb, test_predicate_ub
            )
        
        test_scale = read_csv_data(sources[AFFINEMAP_INIT][SCALE_ID])
        test_offset = read_csv_data(sources[AFFINEMAP_INIT][OFFSET_ID])
        
        am_result = test_star.affine_map(test_scale, test_offset)
        
        self.assertEqual(am_result.get_V().all(), test_am_output.all())


if __name__ == '__main__':
    unittest.main()
