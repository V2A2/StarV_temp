import unittest
import numpy as np
import sys, os
import mat73

from test_inputs.sources import *

os.chdir("../../../../")
print(os.getcwd())
sys.path.insert(0, "engine/nn/layers/signlayer")
from signlayer import *

def read_csv_data(path):        
    return np.array(list(mat73.loadmat(path).values())[0])

class TestSignLayer_evaluate(unittest.TestCase):
    
    """
        Tests SignLayer evaluate method
    """
    
    def test_cstyle_layer(self):
        test_name = 'sign_1'
        test_mode = 'polar_zero_to_pos_one'
        test_num_inputs = 1
        test_input_names = ['in']
        test_num_outputs = 1
        test_output_names = ['out']
        
        sign_layer = SignLayer(test_name, test_mode, test_num_inputs,\
                                  test_input_names, test_num_outputs,\
                                  test_output_names)
        
        input = np.random.rand(3,3) - 0.5
        input[1:3] = 0

        result = sign_layer.evaluate(input)
            
if __name__ == '__main__':
    unittest.main()