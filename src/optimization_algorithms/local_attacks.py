#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
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
                        lib           = "default",
                        algo          = "PGD",
                        enable_speculative_search = False,
                        t_stall       = 0,
                        verbose_iterations = 0,
                        ):

    obj = lambda x: f(Phi(x))
    if verbose_iterations > 0: print("optim_local_attacks from obj value = {:>+9.3E}".format(obj(x_0)))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["attack radius"], is_header=True)
    v_sum = 0
    t_sum = 0
    converged = False; nb_stall_iters = 0; max_stall_iters = max(np.inf,Phi.n+1)

    zero = torch.zeros_like(x_0)

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r = r_0; s = "init"
    history = fill_history(history, x, o, k, t, v, s, additional=[r])
    searches_counter = 0
    d_speculative = zero

    while not(converged):

        k += 1

        if enable_speculative_search and not(torch.equal(d_speculative, zero)):
            altered_line_search_iterator = altered_line_search_step(x, d_speculative)
            dL, vL, tL = evaluate_batch_directions(x, altered_line_search_iterator, obj, t_stall = t_stall)
        else:
            dL = zero
            vL = 0
            tL = 0

        if obj(x+dL) > obj(x):

            x += dL
            r *= 1
            tA = 0
            tS = 0
            vA = 0
            vS = 0
            s = "speculative"
            d_speculative = dL

        else:

            local_attack_step_iterator = local_attack_step(Phi, x, df(Phi(x)), r, r_min=r_min, r_max=r_max,
                                                           r_mult_list = [1.1, 1],
                                                           lib=lib, algo=algo)
            dA, vA, tA = evaluate_batch_directions(x, local_attack_step_iterator, obj, opportunistic=True)

            rA = torch.linalg.norm(dA, ord=float("inf"))
            if obj(x+dA) > obj(x):

                x += dA
                r = min(r_max, max(r_min, rA))
                nb_stall_iters = 0
                tS = 0
                vS = 0
                s = "attack"
                d_speculative = dA

            else:

                search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter), r_max=r_max, empty_search=True, light_search=True)
                dS, vS, tS = evaluate_batch_directions(x, search_step_iterator, obj)
                searches_counter += 1

                if obj(x+dS) > obj(x):

                    x += dS
                    r /= 1.1
                    nb_stall_iters = 0
                    s = "search"
                    d_speculative = dS

                else:

                    r /= 1.5
                    nb_stall_iters += 1
                    s = "failure"
                    d_speculative = zero

        o = obj(x)
        t = tL+tA+tS; t_sum += t
        v = vL+vA+vS; v_sum += v
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
        print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}".format(k, o, r, v_sum))
        print()

    return history
