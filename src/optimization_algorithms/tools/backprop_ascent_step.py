#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Line search with slightly altered directions
# For each r in r_mult_list, compute an alteration d_altered of d and then we test x+r*d_altered

def compute_grad_ascent(df, Phi, x):
    x_var = x.clone().requires_grad_(True)
    phi_x = Phi(x_var)
    with torch.no_grad(): grad_f = df(phi_x.detach())
    (phi_x * grad_f).sum().backward()
    ascent_dir = x_var.grad.detach()
    return ascent_dir

def compute_normalized_grad_ascent(df, Phi, x, r):
    ascent_dir = compute_grad_ascent(df, Phi, x)
    dir_norm = torch.linalg.norm(ascent_dir, ord=float("inf"))
    return ascent_dir * r / dir_norm

def backprop_ascent_step(df, Phi, x, r, r_mult_list=[3/2, 1, 2/3]):
    return iter(compute_normalized_grad_ascent(df, Phi, x, r*rm) for rm in r_mult_list)
