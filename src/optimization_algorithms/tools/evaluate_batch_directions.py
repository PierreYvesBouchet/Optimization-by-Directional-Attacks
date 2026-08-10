#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Evaluation of a batch of directions to identify ascent ones, if any. Returns the best candidate resulting from the trial directions

def evaluate_batch_directions(x, o, directions_iterator, obj, opportunistic=True, skip=False):
    x_best = x.clone().detach(); o_best = o
    stop = skip
    while not(stop):
        try:
            d = next(directions_iterator)
        except StopIteration:
            d = torch.zeros_like(x)
            stop = True
        except Exception as e:
            print("evaluate_batch_directions: exception raised while evaluating a trial direction: {}".format(e))
            d = torch.zeros_like(x)
            stop = True
        if not(stop):
            x_t = x+d
            o_t = obj(x_t)
            if o_t > o_best:
                x_best = x_t
                o_best = o_t
                if opportunistic: stop = True
    return x_best, o_best
