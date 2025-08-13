#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import time



#%% Evaluation of a batch of directions to identify ascent ones, if any
# (t_stall is used to make calls to obj costly to run)

def evaluate_batch_directions(x, directions_iterator, obj, t_stall = 0, opportunistic=True, skip=False):
    best = torch.zeros_like(x); o_best = obj(x)
    number_trial = 0
    stop = skip
    t_in = time.perf_counter()
    while not(stop):
        try:
            d = next(directions_iterator)
        except:
            d = torch.zeros_like(x)
            stop = True
        if not(stop):
            number_trial += 1
            time.sleep(t_stall)
            o_d = obj(x+d)
            if o_d > o_best:
                best = d
                o_best = o_d
                if opportunistic: stop = True
    t_out = time.perf_counter()
    runtime = t_out-t_in
    return best, number_trial, runtime
