#!/usr/bin/python3
import numpy as np

class LogSig_X_TanSig:
    # logsig(x) (*) tansig(y), wher (*) is the Hadamard product, an element-wise product.

    def evaluate(x):
        return np.tanh(y) / (1 + np.exp(-x))
    