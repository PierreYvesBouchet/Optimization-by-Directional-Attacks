#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import torch
import numpy as np

# Useful technical functions
from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction
from src.optimization_algorithms.tools.random_rotation       import random_rotation



#%% Poll step from DSM

def poll_step(x, r, poll_type=["n+1", "2n"][1], empty_poll=False, rng=None):
    if empty_poll: return iter([])
    else:
        n = len(x)
        d = random_unit_direction(n, norm=2, rng=rng)
        dt = d.clone().detach(); dt.resize_(n, 1)
        H = torch.eye(n)-2*dt*d
        if   poll_type == "2n":  directions = [h for h in H]+[-h for h in H]
        elif poll_type == "n+1": directions = [random_rotation(h, theta_max=np.pi/6, rng=rng) for h in H]+[random_rotation(sum([-h for h in H]), theta_max=np.pi/6, rng=rng)]
        else:                    directions = [h for h in H]+[-h for h in H]
        directions = [r*d/torch.linalg.norm(d, ord=float("inf")) for d in directions]
        return iter(directions)
