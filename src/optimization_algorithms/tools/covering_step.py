#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction



#%% Covering step from DSM (simplified implementation)

def covering_step(x, r_covering = 1, empty_covering=False):
    if empty_covering: D = iter([])
    else:
        n = len(x)
        r = (torch.rand(1)*r_covering)**(1/len(x))
        d = random_unit_direction(n)*r
        D = iter([d])
    return D
