#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction



#%% Covering step from DSM (simplified implementation)

def covering_step(x, r_covering=1, nb_directions=1, empty_covering=False, rng=None):
    if empty_covering: return iter([])
    else             : return iter(random_unit_direction(len(x), rng=rng) * (torch.rand(1, generator=rng)*r_covering)**(1/len(x)) for _ in range(nb_directions))
