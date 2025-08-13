#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
import random



#%% Rotate a vector by a random angle of at most theta_max

def random_rotation(v, theta_max=np.pi/4):
    n = len(v)
    R = torch.zeros(n,n)
    I = [i for i in range(n)]
    random.shuffle(I)
    for i in range(int(n/2)):
        theta = theta_max*(-1+2*random.random())
        R[i,   i  ] =    np.cos(theta)
        R[i,   i+1] =    np.sin(theta)
        R[i+1, i  ] = -1*np.sin(theta)
        R[i+1, i+1] =    np.cos(theta)
    return torch.matmul(R, v)
