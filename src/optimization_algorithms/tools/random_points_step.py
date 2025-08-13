#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction



#%% Sequence of random points in the sphere of radius r centred at x (in norm inf by default)

def random_points_step(x, r, nb_points = 1):
    n = len(x)
    directions = []
    for i in range(nb_points):
        d = random_unit_direction(n)*r
        directions.append(d)
    return iter(directions)
