#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# NN attack packages
from src.attack_algorithms.tools.gateway_torchattacks import gateway_torchattacks
# from src.attack_algorithms.tools.gateway_foolbox      import gateway_foolbox

# Package to copy the NN model
##BUG: for some NN (eg., barycenter_into_resnet/Phi.pt), the first attack
#       changes something in the NN: for all x, v = Phi(x) called before first
#       attack differs from v = Phi(x) called after the first attack,
#       even though x is unchanged, and apparently Phi also.
#       So we must do a deep copy of the model prior to the attack.
## Seems not to appear anymore.
# import copy



#%% Attack algorithm

# Attacks model from input_tensor with radius r to seek for aimed_output_tensor
def attack(model, input_tensor, aimed_output_tensor, r,
           lib="default", algo="default",
           ):
    # model_copy = copy.deepcopy(model).eval()
    model_copy = model.eval()
    for param in model_copy.parameters(): param.requires_grad = False
    if   lib == "torchattacks": atk = gateway_torchattacks(model_copy, algo=algo, r=r)
    # elif lib == "foolbox":      atk = gateway_foolbox(     model_copy, algo=algo, r=r) #TODO: Check the gateway_foolbox function before uncommenting this
    else:                       atk = gateway_torchattacks(model_copy, algo=algo, r=r)
    input_batch = input_tensor.unsqueeze(0)
    aimed_output_batch = aimed_output_tensor.unsqueeze(0)
    output_batch = atk(input_batch, aimed_output_batch)
    output_tensor = output_batch.squeeze(0)
    return output_tensor
