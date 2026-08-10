#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# NN attack packages
# Torchattacks package https://github.com/Harry24k/adversarial-attacks-pytorch
import torchattacks



#%% Gateway taking as input a model, an attack radius and a chosen targeted attack algorithm
#   Returns a function atk that is used as atk(input_batch, target_batch)

def gateway_torchattacks(model, r=0.5, algo="default"):
    model_copy = model.eval()
    for param in model_copy.parameters(): param.requires_grad = False
    # Match-case structure would be better, but not available in Python < 3.10
    if   algo == "FGSM"     : atk = torchattacks.FGSM(  model_copy, eps=r)
    elif algo == "RFGSM"    : atk = torchattacks.RFGSM( model_copy, eps=r)
    elif algo == "FFGSM"    : atk = torchattacks.FFGSM( model_copy, eps=r)
    # elif algo == "MIFGSM"   : atk = torchattacks.MIFGSM(model_copy, eps=r) # Valid only for image inputs (code expecting 2D inputs)
    # elif algo == "DIFGSM"   : atk = torchattacks.DIFGSM(model_copy, eps=r) # Valid only for image inputs (code expecting 2D inputs)
    # elif algo == "TIFGSM"   : atk = torchattacks.TIFGSM(model_copy, eps=r) # Valid only for image inputs (code expecting 2D inputs)
    # elif algo == "NIFGSM"   : atk = torchattacks.NIFGSM(model_copy, eps=r) # Valid only for image inputs (code expecting 2D inputs)
    elif algo == "BIM"      : atk = torchattacks.BIM(   model_copy, eps=r)
    elif algo == "PGD"      : atk = torchattacks.PGD(   model_copy, eps=r, random_start=False)
    # elif algo == "UPGD"     : atk = torchattacks.UPGD(  model_copy, eps=r, random_start=False) # Valid only for image inputs (code expecting 2D inputs)
    # elif algo == "PGDRS"    : atk = torchattacks.PGDRS( model_copy, eps=r) # Raises error with respect to dimensions of vectors
    # elif algo == "APGD"     : atk = torchattacks.APGD(  model_copy, eps=r) # Target mode not supported
    # elif algo == "TPGD"     : atk = torchattacks.TPGD(  model_copy, eps=r) # Target mode not supported
    else: return gateway_torchattacks(model, r=r, algo="FFGSM")
    atk.set_model_training_mode(model_training=False, batchnorm_training=False, dropout_training=False)
    atk.set_mode_targeted_by_label(quiet=True)
    return atk
