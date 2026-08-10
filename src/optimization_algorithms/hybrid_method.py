#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
import time

from src.optimization_algorithms.tools.fill_history              import fill_history
from src.optimization_algorithms.tools.evaluate_batch_directions import evaluate_batch_directions
from src.optimization_algorithms.tools.sufficient_increase       import sufficient_increase
from src.optimization_algorithms.tools.local_attack_step         import local_attack_step
from src.optimization_algorithms.tools.covering_step             import covering_step
from src.optimization_algorithms.tools.search_step               import search_step
from src.optimization_algorithms.tools.poll_step                 import poll_step
from src.optimization_algorithms.tools.altered_line_search_step  import altered_line_search_step



#%% Optimization based on a hybrid local attack + DSM

def optim_hybrid_method(f, df, Phi, x_0, r_0,
                        r_dsm_min     = 1E-5,
                        r_dsm_max     = float("inf"),
                        r_atk_min     = 1E-5,
                        r_atk_max     = 1E0,
                        nb_points_max = float("inf"),
                        runtime_max   = float("inf"),
                        k_max         = float("inf"),
                        algo          = "FFGSM",
                        enable_search = False,
                        t_stall       = 0,
                        verbose_iterations = 0,
                        seed          = 0
                        ):

    rng = torch.Generator()
    rng.manual_seed(seed)
    Phi.nb_forward_calls = 0

    obj = lambda x: f(Phi(x))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["attack radius", "poll radius", "attack gain"], is_header=True)
    t_sum = 0
    converged = False

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r_atk = r_0; r_dsm = r_0; s = "init"; attack_gain = 0
    history = fill_history(history, x, o, k, t, v, s, additional=[r_atk, r_dsm, attack_gain])
    searches_counter = 0

    if verbose_iterations > 0: print("optim_hybrid_method("+algo+") from obj value = {:>+9.3E} with seed {}".format(o, seed))

    while not(converged):

        k += 1
        v_sum = Phi.nb_forward_calls
        t_in = time.perf_counter()
        y = Phi(x)

        local_attack_step_iterator = local_attack_step(Phi, x, y, df(y), r_atk, r_min = r_atk_min, r_max = r_atk_max, r_mult_list = [1.2, 1, 0.8], algo=algo, rng=rng)
        tA, oA = evaluate_batch_directions(x, o, local_attack_step_iterator, obj)
        rA = torch.linalg.norm(tA-x, ord=float("inf"))
        attack_gain += max(oA - o, 0)

        if sufficient_increase(oA, o, rA, tau=1E-2, force_false=True):

            x = tA
            o = oA
            r_dsm = min(r_dsm_max, 1.3*r_dsm)
            r_atk = min(r_atk_max, max(r_atk_min, r_atk*1.3))
            s = "attack+skipped"

        else:

            if oA > o:

                x = tA
                o = oA
                r_dsm = min(r_dsm_max, 1.1*r_dsm)
                r_atk = min(r_atk_max, max(r_atk_min, r_atk*1.3))
                s = "attack+"

            else:

                r_dsm = r_dsm
                r_atk = max(r_atk_min, r_atk/1.3)
                s = "failure+"

            covering_step_iterator = covering_step(x, r_covering = r_0, rng=rng)
            tC, oC = evaluate_batch_directions(x, o, covering_step_iterator, obj)

            if oC > o:

                x = tC
                o = oC
                r_dsm = r_dsm
                s += "covering"

            else:

                search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter+1), r_max=r_dsm_max, light_search=True, empty_search=not enable_search, rng=rng)
                tS, oS = evaluate_batch_directions(x, o, search_step_iterator, obj, skip=(r_dsm <= r_dsm_min))
                searches_counter += 1

                if oS > o:

                    x = tS
                    o = oS
                    r_dsm = min(r_dsm_max, 2*r_dsm)
                    s += "search"

                else:

                    poll_step_iterator = poll_step(x, r_dsm, poll_type="n+1", rng=rng)
                    tP, oP = evaluate_batch_directions(x, o, poll_step_iterator, obj, skip=(r_dsm <= r_dsm_min))

                    if oP > o:
                        x = tP
                        o = oP
                        r_dsm = min(r_dsm_max, 2*r_dsm)
                        s += "poll"
                    else:
                        r_dsm = max(r_dsm_min, r_dsm/2)
                        s += "failure"

        t_out = time.perf_counter(); t = t_out-t_in; t_sum += t
        v = Phi.nb_forward_calls - v_sum; v_sum += v
        history = fill_history(history, x, o, k, t, v, s, additional=[r_atk, r_dsm, attack_gain])

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, r_atk = {:>7.1E}, r_dsm = {:>7.1E}, v = {:>8d}, t = {:>6.2f}, s = {:s}".format(k, o, r_atk, r_dsm, v_sum, t_sum, s))

        if k >= k_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of iterations\" triggered")

        if v_sum >= nb_points_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"number of evaluated points\" triggered")

        if r_dsm <= r_dsm_min and r_atk <= r_atk_min:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"r_dsm < r_dsm_min and r_atk < r_atk_min\" triggered")

        if t_sum >= runtime_max:
            converged = True
            if verbose_iterations > 0:
                print("stopping criterion \"excessive runtime\" triggered")

    if verbose_iterations > 0:
        print("k = {:>4d}, obj = {:>+9.3E}, r_atk = {:>7.1E}, r_dsm = {:>7.1E}, v = {:>8d}, t = {:>6.2f}, s = {:s}".format(k, o, r_atk, r_dsm, v_sum, t_sum, s))
        print()

    return history
