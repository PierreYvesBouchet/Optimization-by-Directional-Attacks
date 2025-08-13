#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# NN attack packages
# Torchattacks package https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks



#%% Gateway taking as input a model, an attack radius and a chosen targeted attack algorithm
#   Returns a function atk that is used as atk(input_batch, target_batch)

def gateway_torchattacks(model, r=0.5, algo="default"):
    if   algo == "FGSM" : atk = torchattacks.FGSM(model, eps=r)
    elif algo == "PGD"  : atk = torchattacks.PGD( model, eps=r, random_start=False)
    else                : return gateway_torchattacks(model, r=r, algo="FGSM")
    atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
    atk.set_mode_targeted_by_label(quiet=True)
    return atk
