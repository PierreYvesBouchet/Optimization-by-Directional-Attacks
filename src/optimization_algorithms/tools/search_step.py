#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

from src.optimization_algorithms.tools.poll_step import poll_step
from src.optimization_algorithms.tools.random_line_search_step import random_line_search_step



#%% Search step from DSM

def search_step(x, r, r_max=1E2, empty_search=False, light_search=True):
    if light_search: step = lambda x,r: random_line_search_step(x, r, r_mult_list=[1])
    else           : step = poll_step
    if r > r_max: r = r_max;
    if empty_search: D = iter([])
    else           : D = step(x, r)
    return D
