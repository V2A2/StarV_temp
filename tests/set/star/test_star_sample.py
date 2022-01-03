#!/usr/bin/python3
import sys
import os
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star

def main():
    # dimension of polytome
    dim = 2
    # number of constraints in polytope
    N = 4
    A = np.random.rand(N, dim)

    # compute the convex hull
    P = pc.qhull(A)
    print(P.bounding_box)

    # convert polytope to star
    V = np.array([[0, 1, 0], [0, 0, 1]])
    S = Star(V, P.A, P.b.reshape(-1, 1))
    print(S.__repr__())

    X = S.sample(200)
    print(X)

    P.plot()
    for i in range(X.shape[1]):
        plt.plot(X[0, i], X[1, i], 'go')
    plt.show()

if __name__ == '__main__':
    main()   