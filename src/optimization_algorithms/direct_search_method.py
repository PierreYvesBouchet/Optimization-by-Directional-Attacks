#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np

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
                               enable_speculative_search = False,
                               t_stall       = 0,
                               verbose_iterations = 0,
                               ):

    obj = lambda x: f(Phi(x))
    if verbose_iterations > 0: print("optim_direct_search_method from obj value = {:>+9.3E}".format(obj(x_0)))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["poll radius"], is_header=True)
    v_sum = 0
    t_sum = 0
    converged = False

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
            tC = 0
            tS = 0
            tP = 0
            vC = 0
            vS = 0
            vP = 0
            s = "speculative"
            d_speculative = dL

        else:

            covering_step_iterator = covering_step(x, r_covering = r_0)
            dC, vC, tC = evaluate_batch_directions(x, covering_step_iterator, obj)

            if obj(x+dC) > obj(x):

                x += dC
                r *= 1
                tS = 0
                tP = 0
                vS = 0
                vP = 0
                s = "covering"
                d_speculative = dC

            else:

                search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter+1), r_max=r_max, light_search=True, empty_search=False)
                dS, vS, tS = evaluate_batch_directions(x, search_step_iterator, obj)
                searches_counter += 1

                if obj(x+dS) > obj(x):

                    x += dS
                    r *= 1
                    tP = 0
                    vP = 0
                    s = "search"
                    d_speculative = dS

                else:

                    poll_step_iterator = poll_step(x, r)
                    dP, vP, tP = evaluate_batch_directions(x, poll_step_iterator, obj)

                    if obj(x+dP) > obj(x):
                        x += dP
                        r = min(r_max, 2*r)
                        s = "poll"
                        d_speculative = dP
                    else:
                        r = max(r_min, r/2)
                        s = "failure"
                        d_speculative = zero

        o = obj(x)
        t = tL+tC+tS+tP; t_sum += t
        v = vL+vC+vS+vP; v_sum += v
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
