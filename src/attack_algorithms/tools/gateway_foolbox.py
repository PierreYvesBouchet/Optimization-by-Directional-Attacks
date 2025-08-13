#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TODO: verify compatibility with current version of the code



#%% Libraries import

# NN attack packages
# Foolbox package https://github.com/bethgelab/foolbox
# import foolbox



#%% Gateway taking as input a model, an attack radius and a chosen targeted attack algorithm
#   Returns a function atk that is used as atk(input_batch, target_batch)

# def gateway_foolbox(Phi_x, r=0.5, algo="default"):
#     fmodel = foolbox.PyTorchModel(Phi_x, bounds=(0,1))
#     if algo == "PGD": atk_algo = foolbox.attacks.LinfPGD()
#     atk = lambda d,u: atk_algo(fmodel, d, u, epsilons=[r])[1]
#     return atk
