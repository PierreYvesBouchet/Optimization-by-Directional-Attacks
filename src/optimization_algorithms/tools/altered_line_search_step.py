#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# generic Python packages
import numpy as np

# Useful technical functions
from src.optimization_algorithms.tools.line_search_trial_directions import line_search_trial_directions
from src.optimization_algorithms.tools.random_rotation import random_rotation



#%% Line search with slightly altered directions
# For each r in r_mult_list, compute an alteration d_altered of d and then we test x+r*d_altered

def altered_line_search_step(x, d, r_mult_list=[3/2, 1, 2/3]):
    return iter(line_search_trial_directions(random_rotation(d, theta_max=0*np.pi/8), r_mult_list=r) for r in r_mult_list)
