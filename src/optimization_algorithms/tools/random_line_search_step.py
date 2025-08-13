#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction
from src.optimization_algorithms.tools.line_search_trial_directions import line_search_trial_directions



#%% Random line search

def random_line_search_step(x, r, nb=1, r_mult_list=[2, 1, 1/2, 1/4]):
    n = len(x)
    directions = []
    for i in range(nb):
        d = random_unit_direction(n)*r
        directions += line_search_trial_directions(d, r_mult_list=r_mult_list)
    return iter(directions)
