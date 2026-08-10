#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
import time

from src.optimization_algorithms.tools.fill_history              import fill_history
from src.optimization_algorithms.tools.evaluate_batch_directions import evaluate_batch_directions
from src.optimization_algorithms.tools.covering_step             import covering_step
from src.optimization_algorithms.tools.search_step               import search_step
from src.optimization_algorithms.tools.poll_step                 import poll_step
from src.optimization_algorithms.tools.altered_line_search_step  import altered_line_search_step



#%% Optimization based on the cDSM

def optim_direct_search_method(f, df, Phi, x_0, r_0,
                               r_min         = 1E-5,
                               r_max         = float("inf"),
                               nb_points_max = float("inf"),
                               runtime_max   = float("inf"),
                               k_max         = float("inf"),
                               enable_search = False,
                               enable_speculative_search = False,
                               t_stall       = 0,
                               verbose_iterations = 0,
                               seed          = 0
                               ):

    rng = torch.Generator()
    rng.manual_seed(seed)
    Phi.nb_forward_calls = 0

    obj = lambda x: f(Phi(x))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["poll radius"], is_header=True)
    t_sum = 0
    converged = False

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r = r_0; s = "init"
    history = fill_history(history, x, o, k, t, v, s, additional=[r])
    searches_counter = 0

    if verbose_iterations > 0: print("optim_direct_search_method from obj value = {:>+9.3E} with seed {}".format(o, seed))

    while not(converged):

        k += 1
        v_sum = Phi.nb_forward_calls
        t_in = time.perf_counter()

        covering_step_iterator = covering_step(x, r_covering = r_0, rng=rng)
        tC, oC = evaluate_batch_directions(x, o, covering_step_iterator, obj)

        if oC > o:

            x = tC
            o = oC
            r *= 1
            s = "covering"

        else:

            search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter+1), r_max=r_max, light_search=True, empty_search=not enable_search, rng=rng)
            tS, oS = evaluate_batch_directions(x, o, search_step_iterator, obj)
            searches_counter += 1

            if oS > o:

                x = tS
                o = oS
                r *= 1
                s = "search"

            else:

                poll_step_iterator = poll_step(x, r, rng=rng)
                tP, oP = evaluate_batch_directions(x, o, poll_step_iterator, obj)

                if oP > o:
                    x = tP
                    o = oP
                    r = min(r_max, 2*r)
                    s = "poll"
                else:
                    r = max(r_min, r/2)
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

        if r <= r_min:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"r < r_min\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, r = {:>7.1E}, v = {:>8d}, t = {:>6.2f}".format(k, o, r, v_sum, t_sum))
        print()

    return history
