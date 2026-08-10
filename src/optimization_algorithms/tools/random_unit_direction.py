#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
from botorch.utils.sampling import sample_hypersphere



#%% Random direction in the unit sphere of IR^n

def random_unit_direction(n, norm=float("inf"), rng=None):
    d = sample_hypersphere(n, seed=rng.seed() if rng is not None else None)[0]
    return d/torch.linalg.norm(d, ord=norm)
