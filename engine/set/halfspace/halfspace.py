#!/usr/bin/python3
import numpy as np


class HalfSpace:
    """
    HalfSpace class defining Gx <= g
    """

    def __init__(self, G, g):
        """
        Constructor

        Args:
            G: half-space matrix
            g: half-space vector
        """
        # half-space matrix, half-space vector
        [self.G, self.g] = [np.array([]), np.array([])]
        self.dim = 0  # dimension of half-space

        [n1, m1] = G.shape
        [n2, m2] = [g.shape[0], 1]

        assert n1 == n2, 'error: Inconsistent dimension between half-space matrix and half-space vector'

        assert m2 == 1, 'error: Half-space vector should have one row'

        self.G = G
        self.g = g
        self.dim = m1

    def __repr__(self):
        return "class: %s \nG: %s \ng: %s \ndim: %s" % (self.__class__, self.G,
                                                        self.g, self.dim)

    def contains(self, x):
        """
        check contain

        Args:
            @x: input vector
        
        Returns:
            @bool: = 1 -> half-space contain point x
                   = 0 -> half-space does not contain point x
        """

        # print("\n x ------------------------ \n", x)
        # print("\n self.g ------------------------ \n", self.g)
        # print("\n self.G ------------------------ \n", self.G)

        assert isinstance(x, np.ndarray), 'error: input is not a np array'
        [n, m] = x.shape
        # print("\n n ------------------------ \n", n)
        # print("\n m ------------------------ \n", m)

        assert n == self.dim, 'error: Inconsistent dimension between the vector x and the half-space object'

        assert m == 1, 'error: Input vector x should have one column'

        y = self.g - (self.G @ x).flatten()
        # print("\n y ------------------------ \n", y)
        
        map = np.argwhere(y < 0)[:1] # which index has y(i) < 0'
        # print("\n map ------------------------ \n", map)

        if map.size:
            return 0
        else:
            return 1
