#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Points to consider in a line search in direction d (may be ||d|| =/= 1)

def line_search_trial_directions(d, r_mult_list=[2, 1, 1/2, 1/4]): return [d*r for r in r_mult_list]
