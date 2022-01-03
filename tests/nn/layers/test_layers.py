# ----------------- test for Layer Construction function ---------
import numpy as np
import sys
import os

os.chdir('tests/')
sys.path.append("..")

from engine.nn.funcs.poslin import PosLin
from engine.nn.layers.layer import Layer
from engine.set.star import Star

def main():
    # random input set
    Ai = np.matrix('-0.540814703979925 -0.421878816995180;'
                '0.403580749757606 -0.291562729475043;'
                '0.222355769690372 0.164981737653923;'
                '-0.391349781319239 0.444337590813175;'
                '-0.683641719399254 -0.324718758259433')

    bi = np.matrix('0.727693424272787;'
                '0.867244921118684;'
                '0.960905270006411;'
                '0.805859450556812;'
                '0.653599057168295')

    Vi = np.matrix('-1.28142280110204 0.685008254671879;'
                '3.22068720143861 1.48359989341389;'
                '0.468690315965779 -2.32571060511741;'
                '0.349675922629359 -1.27663092336119;'
                '1.79972069619285 3.39872156367377')

    W = np.matrix('1.5 1;'
                '0 0.5')
    b = np.matrix('0.5;'
                '0.5')
    V = np.matrix('0 0;'
                '1 0;'
                '0 1')
    L = Layer(W=W, b=b, f='poslin')
    #print(L.__repr__())
    # ----------------- end the test for Layer Construction function ---------

    # ----------------- test for Layer reach function ---------
    # V, C, d
    I = Star(V.transpose(), Ai, bi)
    I1 = I.affineMap(W, b)
    #print(I1.__repr__())
    I_list = []
    I_list.append(I)
    S = PosLin.reach(I1)
    S = L.reach(I_list, 'exact-star')
    print(S.__repr__())
    # ----------------- end the test for Layer reach function ---------

if __name__ == '__main__':
    main()   