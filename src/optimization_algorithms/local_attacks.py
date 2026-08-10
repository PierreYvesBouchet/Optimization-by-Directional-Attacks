#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
import time

from src.optimization_algorithms.tools.fill_history              import fill_history
from src.optimization_algorithms.tools.evaluate_batch_directions import evaluate_batch_directions
from src.optimization_algorithms.tools.local_attack_step         import local_attack_step
from src.optimization_algorithms.tools.search_step               import search_step
from src.optimization_algorithms.tools.altered_line_search_step  import altered_line_search_step



#%% Optimization based on a sequence of attacks + (optional) small perturbation when fails

def optim_local_attacks(f, df, Phi, x_0, r_0,
                        r_min         = 1E-5,
                        r_max         = 1E1,
                        nb_points_max = float("inf"),
                        runtime_max   = float("inf"),
                        k_max         = float("inf"),
                        algo          = "FFGSM",
                        enable_search = False,
                        enable_speculative_search = False,
                        verbose_iterations = 0,
                        seed          = 0
                        ):

    rng = torch.Generator()
    rng.manual_seed(seed)
    Phi.nb_forward_calls = 0

    obj = lambda x: f(Phi(x))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["attack radius"], is_header=True)
    t_sum = 0
    converged = False; nb_stall_iters = 0; max_stall_iters = max(np.inf,Phi.n+1)

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r = r_0; s = "init"
    history = fill_history(history, x, o, k, t, v, s, additional=[r])
    searches_counter = 0

    if verbose_iterations > 0: print("optim_local_attacks("+algo+") from obj value = {:>+9.3E} with seed {}".format(o, seed))

    while not(converged):

        k += 1
        v_sum = Phi.nb_forward_calls
        t_in = time.perf_counter()
        y = Phi(x)

        # local_attack_step_iterator = local_attack_step(Phi, x, y, df(y), r, r_min=r_min, r_max=r_max, r_mult_list = [1.2, 1], algo=algo, rng=rng)
        local_attack_step_iterator = local_attack_step(Phi, x, y, df(y), r, r_min=r_min, r_max=r_max, algo=algo, rng=rng)
        tA, oA = evaluate_batch_directions(x, o, local_attack_step_iterator, obj)
        rA = torch.linalg.norm(tA-x, ord=float("inf"))

        if oA > o:

            x = tA
            o = oA
            # r = min(r_max, max(r_min, rA, r))
            r = min(r_max, 1.1*r)
            nb_stall_iters = 0
            s = "attack"

        else:

            search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter), r_max=r_max, empty_search=not enable_search, light_search=True, rng=rng)
            tS, oS = evaluate_batch_directions(x, o, search_step_iterator, obj)
            searches_counter += 1

            if oS > o:

                x = tS
                o = oS
                r = max(r_min, r/1.1)
                nb_stall_iters = 0
                s = "search"

            else:

                r = max(r_min, r/1.1)
                nb_stall_iters += 1
                s = "failure"

        t_out = time.perf_counter(); t = t_out-t_in; t_sum += t
        v = Phi.nb_forward_calls-v_sum; v_sum += v
        history = fill_history(history, x, o, k, t, v, s, additional=[r])

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, t = {:>6.2f}, s = {:s}".format(k, o, r, v_sum, t_sum, s))

        if k >= k_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of iterations\" triggered")

        if v_sum >= nb_points_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of evaluated points\" triggered")

        if s == "failure" and r < r_min:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"r < r_min\" triggered")

        if nb_stall_iters > max_stall_iters:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"successive number of failed iterations\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, t = {:>6.2f}".format(k, o, r, v_sum, t_sum))
        print()

    return history
