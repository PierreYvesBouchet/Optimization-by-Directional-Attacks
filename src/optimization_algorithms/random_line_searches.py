#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch

from src.optimization_algorithms.tools.fill_history              import fill_history
from src.optimization_algorithms.tools.evaluate_batch_directions import evaluate_batch_directions
from src.optimization_algorithms.tools.random_line_search_step   import random_line_search_step
from src.optimization_algorithms.tools.altered_line_search_step  import altered_line_search_step



#%% Optimization based on a sequence of line searches in random directions

def optim_random_line_searches(f, df, Phi, x_0, r_0,
                               r_min         = 1E-5,
                               r_max         = float("inf"),
                               nb_points_max = float("inf"),
                               runtime_max   = float("inf"),
                               k_max         = float("inf"),
                               enable_speculative_search = False,
                               t_stall       = 0,
                               verbose_iterations = 0,
                               ):

    obj = lambda x: f(Phi(x))
    if verbose_iterations > 0: print("optim_random_line_searches from obj value = {:>+9.3E}".format(obj(x_0)))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["radius"], is_header=True)
    v_sum = 0
    t_sum = 0
    converged = False; nb_stall_iters = 0

    zero = torch.zeros_like(x_0)

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r = r_0; s = "init"
    history = fill_history(history, x, o, k, t, v, s, additional=[r])
    d_speculative = zero

    while not(converged):

        k += 1

        if enable_speculative_search and not(torch.equal(d_speculative, zero)):

            altered_line_search_iterator = altered_line_search_step(x, d_speculative)
            dL, vL, tL = evaluate_batch_directions(x, altered_line_search_iterator, obj, t_stall = t_stall)
            s = "speculative"

        else:

            random_line_search_iterator = random_line_search_step(x, r, nb=2, r_mult_list=[1.1, 1, 1/1.1])
            dL, vL, tL = evaluate_batch_directions(x, random_line_search_iterator, obj, t_stall = t_stall)

        rL = torch.linalg.norm(dL, ord=float("inf"))

        if obj(x+dL) > obj(x):

            x += dL
            r = min(r_max, max(r_min, rL))
            nb_stall_iters = 0
            s = "linesearch"
            d_speculative = dL

        else:

            nb_stall_iters += 1
            if nb_stall_iters > Phi.n+1:
                r = max(r_min, r/1.3)
            s = "failure"
            d_speculative = zero

        o = obj(x)
        t = tL; t_sum += t
        v = vL; v_sum += v
        history = fill_history(history, x, o, k, t, v, s, additional=[r])

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, s = {:s}".format(k, o, r, v_sum, s))

        if k >= k_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of iterations\" triggered")

        if v_sum >= nb_points_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of evaluated points\" triggered")

        if r <= r_min:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"r < r_min\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}".format(k, o, r, v_sum))
        print()

    return history
