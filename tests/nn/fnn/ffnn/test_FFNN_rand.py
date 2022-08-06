# ------------- test for FFNN_rand function -------------
import unittest
import copy
import numpy as np
import sys

sys.path.insert(0, "engine/nn/funcs/poslin/")
sys.path.insert(0, "engine/nn/funcs/satlin/")
sys.path.insert(0, "engine/nn/layers/layer/")
sys.path.insert(0, "engine/set/star/")
sys.path.insert(0, "engine/nn/fnn/ffnn/")
sys.path.insert(0, "engine/set/halfspace/")

from halfspace import HalfSpace
from FFNN import FFNN
from star import Star
from poslin import PosLin
from satlin import SatLin
from layer import Layer


class TestFFNNrand(unittest.TestCase):
    """
        Tests FFNN_rand function
    """

    def test_FFNN_rand(self):

        neurons = np.array([2, 3, 3, 2])
        print("\n neurons ------------------------ \n", neurons)

        funcs = ['poslin', 'satlin', 'poslin']
        print("\n funcs ------------------------ \n", funcs)

        net = FFNN.rand(neurons, funcs)
        print("\n net ------------------------ \n", net.__repr__())


if __name__ == '__main__':
    unittest.main()

# ------------- End the test for FFNN_rand function -------------
