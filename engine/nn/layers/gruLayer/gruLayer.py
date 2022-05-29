#!/usr/bin/python3
from _typeshed import WriteableBuffer
import numpy as np

from engine.set.rstar import RStar
from engine.nn.funcs.logsig import LogSig
from engine.nn.funcs.tansig import TanSig

class GRULayer:
    # gruLayer is a fully GRU (gated recurrent unit) layer class that contains reachability analyssi method using
    # rstars
    # author: Sung Woo Choi
    # date: 12/06/2021

    # x: input vector
    # h: output vector
    # c: candidate activation vector (\tilde{h}) (cell state)
    # z: update gate vector 
    # r: reset gate vector (relevance gate)
    # W, U, and b: parameter matrices and vector

    #

    def __init__(obj,
        Wz = np.array([]),     # Weight matrix for update gate, x[t]
        Uz = np.array([]),     # Weight matrix for update gate, h[t-1]
        bz = np.array([]),     # Bias vector for update gate
        Wr = np.array([]),     # Weight matrix for reset (relevance) gate, x[t]
        Ur = np.array([]),     # Weight matrix for reset (relevance) gate, h[t-1]
        br = np.array([]),     # Bias vector for reset (relevance) gate
        Wh = np.array([]),     # Weight matrix for output, x[t]
        Uh = np.array([]),     # Weight matrix for output, h[t-1]
        bh = np.array([])):    # Bias vector for out

        # W = np.random.random((output_features, input_features))
        # U = np.random.random((output_features, output_features))
        # b = np.random.random((output_features,))

        assert isinstance(Wz, np.ndarray), 'error: Wz is not an ndarray'
        assert isinstance(Uz, np.ndarray), 'error: Uz is not an ndarray'
        assert isinstance(bz, np.ndarray), 'error: bz is not an ndarray'
        assert isinstance(Wr, np.ndarray), 'error: Wr is not an ndarray'
        assert isinstance(Ur, np.ndarray), 'error: Ur is not an ndarray'
        assert isinstance(br, np.ndarray), 'error: br is not an ndarray'
        assert isinstance(Wh, np.ndarray), 'error: Wh is not an ndarray'
        assert isinstance(Uh, np.ndarray), 'error: Uh is not an ndarray'
        assert isinstance(bh, np.ndarray), 'error: bh is not an ndarray'

        assert Wz.shape[0] == obj.bz and Wz.shape[1] == Uz.shape[1], 'error: inconsistent dimensions between weight matrices (Wz & Uz) of update gate and its bias vector (bz)'
        assert Wr.shape[0] == obj.br and Wr.shape[1] == Ur.shape[1], 'error: inconsistent dimensions between weight matrices (Wr & Ur) of update gate and its bias vector (br)'
        assert Wh.shape[0] == obj.bh and Wh.shape[1] == Uh.shape[1], 'error: inconsistent dimensions between weight matrices (Wh & Uh) of update gate and its bias vector (bh)'
 
        # update gate
        obj.Wz = Wz     # input state
        obj.Uz = Uz     # hidden state
        obj.bz = bz

        # reset gate
        obj.Wr = Wr     # input state
        obj.Ur = Ur     # hidden state
        obj.br = br

        # output gate
        obj.Wh = Wh     # input state
        obj.Uh = Uh     # hidden state
        obj.bh = bh

        obj.nH = obj.Wh.shape[0]
        obj.nI = obj.W.shape[1]
        return 


    # Evaluate method
    def evaluate(obj, x):
        # evaluation of this layer with a specific vector
        # @x: an input sequence of length n, x[i] is the input at time step i
        # return h: hidden state sequence of length n (output vector)

        # author: Sung Woo Choi
        # date: 12/06/2021

        # z[t] = sigmoid(Wz * [h[t-1], x[t]])
        # r[t] = sigmoid(Wr * [h[t-1], x[t]])
        # c[t] = tanh(W * [r[t] * h[t-1], x[t]])
        # h[t] = (1 - z[t]) * h[t-1] + z[t]*c[t]

        # or

        # z[t] = sigmoid(Wz * x[t] + Uz * h[t-1] + bz)
        # r[t] = sigmoid(Wr * x[t] + Ur * h[t-1] + br)
        # c[t] =    tanh(Wh * x[t] + Uh * (r[t] @ h[t-1]) + bh), where @ is the Hadamard product
        # h[t] = (1 - z[t]) @ h[t-1] + z[t] @ c[t]

        [m, n] = x.shape

        # hidden state
        h = np.zeros(obj.nH, n)
        r = np.zeros(obj.nH, n)
        z = np.zeros(obj.nH, n)
        c = np.zeros(obj.nH, n)

        Wzx = obj.Wz @ x
        Wrx = obj.Wr @ x
        Whx = obj.Wx @ x

        # * is Hadamard product
        for t in range(n):
            if t == 0:  # h[0] = 0
                z[t] = LogSig.evaluate(Wzx[:, t] + obj.bz)
                r[t] = LogSig.evaluate(Wrx[:, t] + obj.br)
                c[t] = TanSig.evaluate(Whx[:, t] + obj.Uh @ obj.bh)
                h[:, t] = z[t] * c[t]
            else:
                z[:, t] = LogSig.evaluate(Wzx + obj.Uz @ h[:, t-1] + obj.bz)
                r[:, t] = LogSig.evaluate(Wrx + obj.Ur @ h[:, t-1] + obj.br)
                c[:, t] = TanSig.evaluate(Whx + obj.Uh @ (r[:, t] * h[:, t-1]) + obj.bh)
                h[:, t] = (1 - z[:, t]) * h[:, t-1] + z[:, t] * c[:, t]
        return h


    def reach(obj, 
                I,
           method,
           option,
          dis_opt,
        lp_solver = ''):

        assert isinstance(I, RStar), 'error: input set is not relaxed star (RStar)'

        n = len(I)
        O = [] # output reachable set sequence

        # update gate
        Wz1 = obj.Wz
        Uz1 = obj.Uz
        bz1 = obj.bz
        # reset gate
        Wr1 = obj.Wr
        Ur1 = obj.Ur
        br1 = obj.br
        # output gate
        Wh1 = obj.Wh
        Uh1 = obj.Uh
        bh1 = obj.bh


        WzI = []
        WrI = []
        WhI = []
        for t in range(n):
            WzI.append(I[t].affineMap(Wz1, bz1))
            WrI.append(I[t].affineMap(Wr1, br1))
            WhI.append(I[t].affineMap(Wh1, Uh1 @ bh1))

        Z, R, C, H = [], [], [], []
        for t in range(n):
            if t == 0:
                Z.append(LogSig.reach(WzI[t]))
                R.append(LogSig.reach(WrI[t]))
                C.append(LogSig.reach(WhI[t]))
                H.append(Z.product(C))
            else:
                Z.append(LogSig.reach(WzI[t].Sum(H[t-1].affineMap(Uz1, bz1))))
                R.append(LogSig.reach(WrI[t].Sum(H[t-1].affineMap(Ur1, br1))))
                T = R[t].Product(H[t-1])
                C.append(LogSig.reach(WhI[t].Sum(T.affineMap(Uh1, bh1))))
                H.append(Z.Product(C))









