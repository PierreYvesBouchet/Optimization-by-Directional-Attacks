#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries import

import torch
import time

from src.optimization_algorithms.tools.fill_history import fill_history
from src.optimization_algorithms.tools.backprop_ascent_step import backprop_ascent_step
from src.optimization_algorithms.tools.evaluate_batch_directions import evaluate_batch_directions


#%% Optimization based on steepest ascent (unconstrained) using backpropagation to compute the gradient

def optim_backprop_ascent(f, df, Phi, x_0, r_0,
                           r_min              = 1E-5,
                           r_max              = float("inf"),
                           nb_points_max      = float("inf"),
                           runtime_max        = float("inf"),
                           k_max              = float("inf"),
                           verbose_iterations = 0,
                           seed               = 0
                           ):

    rng = torch.Generator()
    rng.manual_seed(seed)
    Phi.nb_forward_calls = 0

    def obj(x):
        with torch.no_grad(): return f(Phi(x))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status",
                           additional=["radius"], is_header=True)
    t_sum = 0
    converged = False

    x = x_0.clone().detach()
    o = obj(x)
    k = 0; t = 0; v = 1; r = r_0; s = "init"
    history = fill_history(history, x.clone(), o, k, t, v, s, additional=[r])

    if verbose_iterations > 0:
        with torch.no_grad(): print("optim_backprop_ascent from obj value = {:>+9.3E} with seed {}".format(o, seed))

    while not converged:

        k += 1
        v_sum = Phi.nb_forward_calls
        t_in = time.perf_counter()

        backprop_ascent_step_iterator = backprop_ascent_step(df, Phi, x, r, r_mult_list=[1])
        tB, oB = evaluate_batch_directions(x, o, backprop_ascent_step_iterator, obj)

        if oB > o:

            x = tB
            o = oB
            r = min(r_max, r * 1.1)
            s = "backprop"

        else:

            r = max(r_min, r / 1.1)
            s = "failure"

        t_out = time.perf_counter(); t = t_out-t_in; t_sum += t
        v = Phi.nb_forward_calls-v_sum
        v_sum += v
        history = fill_history(history, x.clone(), o, k, t, v, s, additional=[r])

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, t = {:>6.2f}, s = {:s}".format(k, o, r, v_sum, t_sum, s))

        if k >= k_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"number of iterations\" triggered")

        if v_sum >= nb_points_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"number of evaluated points\" triggered")

        if r <= r_min:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"r < r_min\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0: print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, t = {:>6.2f}".format(k, o, r, v_sum, t_sum))
        print()

    return history
