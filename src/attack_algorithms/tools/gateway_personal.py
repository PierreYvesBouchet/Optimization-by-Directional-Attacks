#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import itertools
import random
from src.optimization_algorithms.tools.random_unit_direction import random_unit_direction



#%% Gateway taking as input a model, an attack radius and a chosen targeted attack algorithm
#   Returns a function atk that is used as atk(input_batch, target_batch)
#   The function atk takes an input batch and a target batch as arguments and returns an adversarial batch.

def gateway_personal(model, r=0.5, algo="diagonals"):

    def square_loss(t1, t2): return torch.linalg.norm(t1 - t2)

    # Tests diagonal attacks (d = [+-r, +-r, ..., +-r]) in a greedy way (changing each dimension one time, in order, keeping the best sign among the two) and returns the first one we find that increases the loss of the model. If none is found, returns [0,0,...,0].
    def atk_diagonals(input_batch, target_batch, nb_test=10):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss_0 = square_loss(model(input_tensor), target_tensor)
        d = torch.ones_like(input_tensor) * r
        loss_current = square_loss(model(input_tensor + d), target_tensor)
        I = [i for i in range(input_tensor.numel())]
        random.shuffle(I)
        for i in I[:nb_test]:
            d_changed = d.clone(); d_changed[i] = -d_changed[i]; input_adv_changed = input_tensor + d_changed; loss_changed = square_loss(model(input_adv_changed), target_tensor)
            if loss_changed < loss_current: d = d_changed; loss_current = loss_changed
        if loss_current < loss_0: return input_tensor + d
        return torch.zeros_like(input_batch)

    # # Tests diagonal attacks (d = [+-r, +-r, ..., +-r]) in a greedy way (changing each dimension one time, in order, keeping the best sign among the two) and returns the first one we find that increases the loss of the model. If none is found, returns [0,0,...,0].
    # def atk_diagonals(input_batch, target_batch, len_subdiv=10):
    #     input_tensor = input_batch.squeeze(0)
    #     target_tensor = target_batch.squeeze(0)
    #     dim = input_tensor.numel()
    #     loss_0 = square_loss(model(input_tensor), target_tensor)
    #     d = (2*(torch.rand(dim) < 0.5)-1) * r
    #     loss_current = square_loss(model(input_tensor + d), target_tensor)
    #     I = [i for i in range(dim)]
    #     lenI = len(I)
    #     random.shuffle(I)
    #     for i in range(dim // len_subdiv):
    #         J = [I.pop() for _ in range(min(len_subdiv, lenI))]
    #         lenI -= len(J)
    #         d_changed = d.clone()
    #         d_changed[J] *= -1
    #         input_adv_changed = input_tensor + d_changed
    #         loss_changed = square_loss(model(input_adv_changed), target_tensor)
    #         if loss_changed < loss_current: d = d_changed; loss_current = loss_changed
    #     if loss_current < loss_0: return input_tensor + d
    #     return torch.zeros_like(input_batch)


    # MC-FGSM with random subset of canonical directions
    def atk_mc_fgsm_canonical(input_batch, target_batch, nb_dims=5):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss_0 = square_loss(model(input_tensor), target_tensor)
        grad = torch.zeros_like(input_tensor)
        I = [i for i in range(input_tensor.numel())]
        random.shuffle(I)
        basis = torch.eye(input_tensor.numel())
        for i in I[:nb_dims]:
            loss_d = square_loss(model(input_tensor + r*basis[i]), target_tensor)
            grad[i] = (loss_d - loss_0) / r
        adv = input_tensor - r * torch.sign(grad)
        return adv

    # MC-FGSM with random unitary directions
    def atk_mc_fgsm_random(input_batch, target_batch, nb_dims=20):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss_0 = square_loss(model(input_tensor), target_tensor)
        grad = torch.zeros_like(input_tensor)
        for _ in range(nb_dims):
            d = random_unit_direction(input_tensor.numel())
            loss_d = square_loss(model(input_tensor + r*d), target_tensor)
            grad += (loss_d - loss_0) / r * d
        adv = input_tensor - r * torch.sign(grad)
        return adv


    def atk_grad_simplex_reduced(input_batch, target_batch, nb_dims=10):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss_0 = square_loss(model(input_tensor), target_tensor)
        grad = torch.zeros_like(input_tensor)
        I = [i for i in range(input_tensor.numel())]
        random.shuffle(I)
        nb_success = 0
        for i in I[:nb_dims]:
            d = torch.zeros_like(input_tensor)
            d[i] = r
            loss_d = square_loss(model(input_tensor + d), target_tensor)
            grad[i] = (loss_d - loss_0) / r
        if torch.all(grad == 0): return torch.zeros_like(input_batch)
        grad /= torch.linalg.norm(grad, ord=float("inf"))
        grad *= r
        input_adv = input_tensor - grad.reshape_as(input_tensor)
        return input_adv.squeeze(0)


    # Computes a simplex gradient (canonical) instead of backpropagation, and returns it as an adversarial example. If the gradient is zero, returns [0,0,...,0].
    def atk_grad_simplex(input_batch, target_batch):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss_0 = square_loss(model(input_tensor), target_tensor)
        grad = torch.zeros_like(input_tensor)
        for i in range(input_tensor.numel()):
            d = torch.zeros_like(input_tensor)
            d[i] = r
            loss_d = square_loss(model(input_tensor + d), target_tensor)
            grad[i] = (loss_d - loss_0) / r
        if torch.all(grad == 0): return torch.zeros_like(input_batch)
        grad /= torch.linalg.norm(grad, ord=float("inf"))
        grad *= r
        input_adv = input_tensor - grad.reshape_as(input_tensor)
        return input_adv.squeeze(0)


    # Tests the SimBA attack (https://arxiv.org/abs/1905.07121) and returns the first one we find that increases the loss of the model. If none is found, returns [0,0,...,0].
    def atk_simba(input_batch, target_batch, nb_test=20):
        input_tensor = input_batch.squeeze(0)
        target_tensor = target_batch.squeeze(0)
        loss = square_loss(model(input_tensor), target_tensor)
        d = torch.zeros_like(input_tensor)
        I = [i for i in range(input_tensor.numel())]
        random.shuffle(I)
        for i in I[:nb_test]:
            d[i] = +r; loss_p = square_loss(model(input_tensor+d), target_tensor)
            if loss_p < loss: loss = loss_p
            else:
                d[i] = -r; loss_m = square_loss(model(input_tensor+d), target_tensor)
                if loss_m < loss: loss = loss_m
                else: d[i] = 0.0
        return input_tensor + d


    if algo == "diagonals"   : return atk_diagonals
    if algo == "grad_simplex": return atk_grad_simplex_reduced
    if algo == "mc_fgsm"     : return atk_mc_fgsm_canonical
    if algo == "simba"       : return atk_simba
    if algo == "simba_light" : return lambda input_batch, target_batch: atk_simba(input_batch, target_batch, nb_test=10)
    if algo == "simba_heavy" : return lambda input_batch, target_batch: atk_simba(input_batch, target_batch, nb_test=50)
    raise ValueError("Invalid algorithm specified")
