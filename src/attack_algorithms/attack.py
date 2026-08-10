#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import random
import itertools

# NN attack packages
from src.attack_algorithms.tools.gateway_torchattacks import gateway_torchattacks



#%% Black-box attack algorithms

# For the black-box attacks we implement ourselves
def square_loss(t1, t2, components=[]):
    if len(components) == 0: return torch.linalg.norm(t1 - t2)
    else                   : return torch.linalg.norm(t1[components] - t2[components])

# SimBA attack (https://arxiv.org/abs/1905.07121)
def simba(Phi, input_tensor, target_tensor, r, nb_tests=20):
    loss = square_loss(Phi(input_tensor), target_tensor)
    d = torch.zeros_like(input_tensor)
    y_comp_effective = torch.tensor([i for i in range(target_tensor.numel()) if not(i in Phi.inactive_subspace_f)])
    I = [i for i in range(input_tensor.numel())]
    random.shuffle(I)
    for i in I[:nb_tests]:
        d[i] = +r; loss_p = square_loss(Phi(input_tensor+d), target_tensor, y_comp_effective)
        if loss_p < loss: loss = loss_p
        else:
            d[i] = -r; loss_m = square_loss(Phi(input_tensor+d), target_tensor, y_comp_effective)
            if loss_m < loss: loss = loss_m
            else: d[i] = 0.0
    return d

# Attack picking a random number of components and then tests all variations (+-r on all components k), and returns the best
def custom_bb_attack(Phi, input_tensor, target_tensor, r, alpha=8/10, nb_tests_min=1, nb_tests_max=3, opportunistic=True):
    n = input_tensor.numel()
    m = target_tensor.numel()

    y_comp_effective = torch.tensor([i for i in range(m) if not(i in Phi.inactive_subspace_f)])
    loss = square_loss(Phi(input_tensor), target_tensor, y_comp_effective)
    d = torch.zeros_like(input_tensor)

    nb_tests = nb_tests_min
    while nb_tests < nb_tests_max and random.random() < alpha: nb_tests += 1
    nb_tests = min(nb_tests, n)
    I = [True] * nb_tests + [False] * (n - nb_tests)
    random.shuffle(I)

    dirs_altered = list(itertools.product([-r, r], repeat=nb_tests))
    random.shuffle(dirs_altered)

    for dd in dirs_altered:
        d_changed = d.clone()
        d_changed[I] = torch.tensor([float(u) for u in dd])
        loss_changed = square_loss(Phi(input_tensor+d_changed), target_tensor, y_comp_effective)
        if loss_changed < loss:
            if opportunistic: return d_changed
            else: loss = loss_changed; d = d_changed

    return d



#%% Main attack functions

#  Attacks from input_tensor with radius r to seek for target_tensor
def torchattack_attack(Phi_xr_reduced, input_tensor, target_tensor, r, algo="default"):
    atk = gateway_torchattacks(Phi_xr_reduced, algo=algo, r=r)
    output_tensor = atk(input_tensor, target_tensor)
    return output_tensor

def bb_attack(Phi, input_tensor, target_tensor, r, algo="SimBA"):
    if   algo == "SimBA" : return custom_bb_attack(Phi, input_tensor, target_tensor, r) # simba(Phi, input_tensor, target_tensor, r)
    elif algo == "custom": return custom_bb_attack(Phi, input_tensor, target_tensor, r)
    else: raise ValueError("Unknown black-box attack algorithm: {:s}".format(algo))