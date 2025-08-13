#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import torch
import numpy as np

# Import of the class generating Phi_(x,r) and the function doing targeted attacks
from src.attack_algorithms.Phi_xr_model import Phi_xr_model
from src.attack_algorithms.attack import attack

# Useful technical function
from src.optimization_algorithms.tools.random_rotation import random_rotation



#%% Local attack using the NN Phi_(x,r)

def local_attack_step(Phi, x, dfy, r, r_min = 1E-7, r_max = 1E1,
                      r_mult_list = [1],
                      add_dfy_norm = False, normalize_dfy = False,
                      rotate_attack_dir = False,
                      lib = "default", algo = "default",
                      ):


    # Pre-definition of Phi_(x,r)
    Phi_xr = Phi_xr_model(Phi, x, 1, output_components_ignored=Phi.inactive_subspace_f)

    # Selection of all attack radii
    r_list = [r*m for m in r_mult_list]
    r_list_filtered = []
    i = 0
    while i < len(r_list) and r_list[i] >  r_max: i += 1
    if i > 0: r_list_filtered.append(r_max)
    while i < len(r_list) and r_list[i] >= r_min: r_list_filtered.append(r_list[i]); i += 1
    if i < len(r_list): r_list_filtered.append(r_min)
    if add_dfy_norm: r_list_filtered.append(torch.linalg.norm(dfy)/2)
    if normalize_dfy: dfy /= torch.linalg.norm(dfy, ord=float("inf"))

    # Function doing the attack
    def attack_fct(Phi_xr, x, dfy, ri,
                   lib=lib, algo=algo,
                   rotate_attack_dir=rotate_attack_dir):
        Phi_xr.set_r(ri)
        dfy_comp_removed = dfy[ [i for i in range(len(dfy)) if not(i in Phi_xr.Phi.inactive_subspace_f)] ]
        d0 = Phi_xr.get_reference_for_attack()
        d = attack(Phi_xr, d0, dfy_comp_removed, 0.5, lib=lib, algo=algo)
        d_scaled = Phi_xr.rescale_back_attack_direction(d)
        if rotate_attack_dir: d_scaled = random_rotation(d_scaled, theta_max=np.arccos(1/len(x)))
        return d_scaled

    return iter(attack_fct(Phi_xr, x, dfy, ri) for ri in r_list_filtered)
