#!/usr/bin/python3
import numpy as np

class GRU:
    # GRU class contains method for reachbility analysis for Layer with
    # GRU (gated reccurent units).

    # main function for rechability analysis
    def reach(I,             # an input star set
            method,          # 'approx-star', 'approx-zono', or 'abs-dom'
            option = '',     # = 'parallel' or '' using parallel computation or not
            relaxFactor = 0, # for relaxed approx-star method 
            disp_opt = '', 
            lp_solver = 'gurobi'):

        def reach_rstar_approx(I):
            # @I: input RStar set
            # return: output RStar set
            from engine.set.rstar import RStar

            assert isinstance(I, RStar), 'error: input set is not a RStar set'

            

        def stepGRU_rstar(I, index,):

        if method == 'approx-rstar':
            return GRU.reach_rstar_approx(I)
        else:
            raise Exception('error: unknown or unsupported reachability method for layer with GRU')