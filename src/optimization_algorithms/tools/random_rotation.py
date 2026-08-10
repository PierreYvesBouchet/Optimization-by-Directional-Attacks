#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
import random



#%% Rotate a vector by a random angle of at most theta_max

def random_rotation(v, theta_max=np.pi/4, rng=None):
    n = len(v)
    R = torch.zeros(n,n)
    if rng is None: rng = torch.Generator()
    I = torch.randperm(n, generator=rng).tolist()
    for i in range(int(n/2)):
        theta = theta_max*(-1+2*torch.rand(1, generator=rng).item())
        R[I[i],   I[i]  ] =    np.cos(theta)
        R[I[i],   I[i+1]] =    np.sin(theta)
        R[I[i+1], I[i]  ] = -1*np.sin(theta)
        R[I[i+1], I[i+1]] =    np.cos(theta)
    return torch.matmul(R, v)
