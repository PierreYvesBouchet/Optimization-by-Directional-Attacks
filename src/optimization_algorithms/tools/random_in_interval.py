#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Uniform distribution in [a,b]

def random_in_interval(a, b): return a+torch.rand(1)*(b-a)
