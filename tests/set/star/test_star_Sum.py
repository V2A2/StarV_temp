#!/usr/bin/python3
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

os.chdir('tests/')
sys.path.append("..")

from engine.set.star import Star
from engine.set.rstar import RStar
from engine.nn.funcs.tansig import TanSig

def main():
    np.set_printoptions(precision=25)
    V1 = np.matrix('1, 1, 0; 0, 1, 0; 0, 0, 1')
    C1 = np.matrix('1, 0; -1, 0; 0, 1; 0, -1')
    d1 = np.matrix('1; 1; 1; 1')
    S1 = Star(V1, C1, d1)

    V2 = np.matrix('1, 0; 0, 1; 1, 1')
    C2 = np.matrix('1; -1')
    d2 = np.matrix('0.5; 0.5')
    S2 = Star(V2, C2, d2)

    W = np.matrix('2, 1, 1; 1, 0, 2; 0, 1, 0')
    b = np.matrix('0.5; 0.5; 0')

    S3 = S1.affineMap(W, b)
    
    S12 = S1.Sum(S2)
    S13 = S1.Sum(S3)

    print('\nS1:\n', S1.__repr__())
    print('\nS2:\n', S2.__repr__())
    print('\nS3:\n', S3.__repr__())

    print('\nS12:\n', S12.__repr__())
    print('\nS13:\n', S13.__repr__())

 

# import scipy as sp
# from scipy.spatial import HalfspaceIntersection
# from scipy.spatial import ConvexHull
# import mpl_toolkits.mplot3d as a3
# import matplotlib.colors as colors
# def main():
    # w = np.array([1., 1., 1.])
    # # ∑ᵢ hᵢ wᵢ qᵢ - ∑ᵢ gᵢ wᵢ <= 0 
    # #  qᵢ - ubᵢ <= 0
    # # -qᵢ + lbᵢ <= 0 
    # halfspaces = np.array([
    #                     [ 1.,  0.,  0., -1],
    #                     [ 0.,  1.,  0., -1],
    #                     [ 0.,  0.,  1., -1],
    #                     [-1.,  0.,  0.,  -1],
    #                     [ 0., -1.,  0.,  -1],
    #                     [ 0.,  0., -1.,  -1]
    #                     ])
    # feasible_point = np.array([0., 0., 0.])
    # hs = HalfspaceIntersection(halfspaces, feasible_point)
    # verts = hs.intersections
    # hull = ConvexHull(verts)
    # faces = hull.simplices

    # ax = a3.Axes3D(plt.figure())
    # ax.dist=10
    # ax.azim=30
    # ax.elev=10
    # ax.set_xlim([-3,3])
    # ax.set_ylim([-3,3])
    # ax.set_zlim([-3,3])

    # for s in faces:
    #     sq = [
    #         [verts[s[0], 0], verts[s[0], 1], verts[s[0], 2]],
    #         [verts[s[1], 0], verts[s[1], 1], verts[s[1], 2]],
    #         [verts[s[2], 0], verts[s[2], 1], verts[s[2], 2]]
    #     ]

    #     f = a3.art3d.Poly3DCollection([sq])
    #     f.set_color(colors.rgb2hex(sp.rand(3)))
    #     f.set_edgecolor('k')
    #     f.set_alpha(0.1)
    #     ax.add_collection3d(f)

    # plt.show()


    # W = np.array([[2, 1, 1], [1, 0, 2], [0, 1, 0]])
    # b = np.array([[0.5], [0.5], [0]])

    # print('W: \n', W)
    # print('b: \n', b)
    
    # V1 = np.matrix('0 1 0 0; 0 0 1 0; 0 0 0 1')
    # C1 = np.vstack((np.eye(3), -np.eye(3)))
    # d1 = np.vstack((np.ones((6,1))))
    # # V1 = np.array([[1, 1, 0], [0, 1, 0], [0, 0 ,1]])
    # # C1 = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])
    # # d1 = np.array([[1], [1], [1], [1]])

    # S1 = Star(np.matrix(V1), np.matrix(C1), np.matrix(d1))

    # # V2 = np.array([[1, 0], [0, 1], [1, 1]])
    # # C2 = np.array([[1], [-1]])
    # # d2 = np.array([[0.5], [0.5]])
    # # S2 = Star(np.matrix(V2), np.matrix(C2), np.matrix(d2))

    # S3 = S1.affineMap(np.matrix(W), np.matrix(b))
    # print(S3.__repr__())

    # S3.plot()
    
    # print('S1: \n', S1.__repr__())
    # print('S2: \n', S2.__repr__())
    # print('S3: \n', S3.__repr__())

    # S12 = S1.Sum(S2)
    # S13 = S1.Sum(S3)

    # print('Is S12 an infeasible set? ', S12.isEmptySet())
    # print('Is S13 an infeasible set? ', S13.isEmptySet()) 

    # print('S12: \n', S12.__repr__())
    # print('S13: \n', S13.__repr__())
    # P1 = pc.Polytope(S12.C, np.array(S12.d).reshape(-1))
    # P2 = pc.Polytope(S13.C, np.array(S13.d).reshape(-1))

    # print(P1)
    # print(P2)

    # P1.plot()
    # P2.plot()
    # plt.show()


if __name__ == '__main__':
    main()