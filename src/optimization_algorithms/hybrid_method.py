#!/#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch
import numpy as np
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
                        lib           = "default",
                        algo          = "FGSM",
                        enable_speculative_search = False,
                        t_stall       = 0,
                        verbose_iterations = 0,
                        ):

    obj = lambda x: f(Phi(x))
    if verbose_iterations > 0: print("optim_hybrid_method from obj value = {:>+9.3E}".format(obj(x_0)))

    history = fill_history([], "x", "f(Phi(x))", "k", "runtime", "cache size", "iteration status", additional=["attack radius", "poll radius", "attack gain"], is_header=True)
    v_sum = 0
    t_sum = 0
    converged = False

    zero = torch.zeros_like(x_0)

    x = x_0.clone().detach(); o = obj(x); k = 0; t = 0; v = 0; r_atk = r_0; r_dsm = r_0; s = "init"; attack_gain = 0
    history = fill_history(history, x, o, k, t, v, s, additional=[r_atk, r_dsm, attack_gain])
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
            r_atk *= 1
            r_dsm *= 1
            tA = 0
            tS = 0
            tC = 0
            tP = 0
            vA = 0
            vS = 0
            vC = 0
            vP = 0
            s = "speculative"
            d_speculative = dL

        else:

            local_attack_step_iterator = local_attack_step(Phi, x, df(Phi(x)), r_atk, r_min = r_atk_min, r_max = r_atk_max,
                                                           r_mult_list = [1],
                                                           lib=lib, algo=algo)
            dA, vA, tA = evaluate_batch_directions(x, local_attack_step_iterator, obj)

            rA = torch.linalg.norm(dA, ord=float("inf"))
            attack_gain += max(obj(x+dA) - obj(x), 0)
            if sufficient_increase(obj(x+dA), obj(x), rA, tau=1E-3):

                x += dA
                r_dsm = (r_dsm if r_dsm > r_dsm_min else 2*r_dsm)
                r_atk = min(r_atk_max, max(r_atk_min, r_atk*1.3))
                tS = 0
                tC = 0
                tP = 0
                vS = 0
                vC = 0
                vP = 0
                s = "attack+skipped"
                d_speculative = dA

            else:

                if obj(x+dA) > obj(x):

                    x += dA
                    r_atk = min(r_atk_max, max(r_atk_min, r_atk*1.1))
                    s = "attack+"
                    d_speculative = dA

                else:

                    r_atk = max(r_atk_min, r_atk/1.3)
                    s = "failure+"

                covering_step_iterator = covering_step(x, r_covering = r_0)
                dC, vC, tC = evaluate_batch_directions(x, covering_step_iterator, obj)

                if obj(x+dC) > obj(x):

                    x += dC
                    r_dsm = r_dsm
                    tS = 0
                    tP = 0
                    vS = 0
                    vP = 0
                    s += "covering"
                    d_speculative = dC

                else:

                    search_step_iterator = search_step(x, r_0*np.sqrt(searches_counter+1), r_max=r_dsm_max, light_search=True, empty_search=False)
                    dS, vS, tS = evaluate_batch_directions(x, search_step_iterator, obj, skip=(r_dsm <= r_dsm_min))
                    searches_counter += 1

                    if obj(x+dS) > obj(x):

                        x += dS
                        r_dsm = min(r_dsm_max, 2*r_dsm)
                        tP = 0
                        vP = 0
                        s += "search"
                        d_speculative = dS

                    else:

                        poll_step_iterator = poll_step(x, r_dsm, poll_type="n+1")
                        dP, vP, tP = evaluate_batch_directions(x, poll_step_iterator, obj, skip=(r_dsm <= r_dsm_min))

                        if obj(x+dP) > obj(x):
                            x += dP
                            r_dsm = min(r_dsm_max, 2*r_dsm)
                            s += "poll"
                            d_speculative = dP
                        else:
                            r_dsm = max(r_dsm_min, r_dsm/2)
                            s += "failure"
                            d_speculative = zero

        o = obj(x)
        t = tL+tA+tC+tS+tP; t_sum += t
        v = vL+vA+vC+vS+vP; v_sum += v
        history = fill_history(history, x, o, k, t, v, s, additional=[r_atk, r_dsm])

        # print(vA,vC,vS,vP)

        if verbose_iterations > 0 and k % verbose_iterations == 0:
            print("k = {:>4d}, obj = {:>+9.3E}, r_atk = {:>7.1E}, r_dsm = {:>7.1E}, v = {:>8d}, s = {:s}".format(k, o, r_atk, r_dsm, v_sum, s))

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
        print("k = {:>4d}, obj = {:>+9.3E}, r_atk = {:>7.1E}, r_dsm = {:>7.1E}, v = {:>8d}, s = {:s}".format(k, o, r_atk, r_dsm, v_sum, s))
        print()

    return history
