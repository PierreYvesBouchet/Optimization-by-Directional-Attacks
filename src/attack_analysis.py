#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# General Python-related packages
import time

# Torch-related packages
import torch

# Imports of the default attack algorithm
from src.attack_algorithms.attack import attack
from src.attack_algorithms.Phi_xr_model import Phi_xr_model



#%% Function analyzing the relevance of "attack" algo from "lib" used on Phi_x
#   for the purpose to solve maximize f(Phi(x))
#   measures the attack's runtime and whether or not the attack is successful
#       (in the sense of whether or not f(Phi(x+d)) > f(Phi(x))),
#   at various x for various radii r, always in the direction u = df(Phi(x));
#   then stores the results in attack_analysis(lib;algo).pt located at path_to_save

def attack_analysis(f, df, Phi, X, path_to_save,
                    exp_r_min = -6, exp_r_max = 0,
                    lib = "default", algo = "default", verbose = 0,
                    ):

    # Shorthand for f o Phi
    obj = lambda x: f(Phi(x))

    if verbose >= 1: print("study of {:s} algo from {:s} lib".format(algo, lib))

    # Sorting X by ascending values of f(Phi( ))
    X.sort(key=lambda x: obj(x))
    nb_x = len(X)

    # Radii r considered in the attacks
    # R = [(1Ei, 5Ei) for all i in [[exp_r_min, exp_r_max]]
    R = [i*10**j for j in range(exp_r_min, exp_r_max) for i in [1, 5]]
    nb_r = len(R)

    # List of values f(Phi(x)) for x in X
    O = []

    # Lists per (x,r) in X times R, ordered as [[(x,r) for r in R] for x in X]
    S = [] # successes or not, in the sense bool(f(Phi(x+d)) > f(Phi(x)))
    T = [] # runtimes
    A = [] # f(Phi(x+d))

    # Number of digits in the product nb_x * nb_r
    digits = 0
    while (nb_x*nb_r)/10**digits > 1: digits += 1
    str_digits = "{:>"+str(digits)+"d}/{:<"+str(digits)+"d}"

    # Header for prints
    header_trial = "{:^"+str(2*digits+1)+"s}"; header_trial = header_trial.format("trial")
    header_time  = "{:^8s}".format("time (ms)")
    header_obj_x = "{:^11s}".format("f(Phi(x))")
    header_obj_a = "{:^11s}".format("f(Phi(x+d))")
    header_delta = "{:^11s}".format("delta")
    header_r     = "{:^5s}".format("r")
    header_s     = "{:^7s}".format("success")
    header       = " | ".join([header_trial, header_time, header_obj_x, header_obj_a, header_delta, header_r, header_s])
    str_sep      = "-"*(2*digits+1 + 8 + 11 + 11 + 11 + 5 + 7) + "-"*3*7
    if verbose >= 3: print(header); print(str_sep)

    # Run per (x,r) in X times R; loop as [[(x,r) for r in R] for x in X]
    for i in range(nb_x):

        x = X[i]; yx = Phi(x); ox = obj(x)
        O.append(ox)
        S_x = []
        T_x = []
        A_x = []
        Phi_xr = Phi_xr_model(Phi, x, 1)

        for j in range(nb_r):

            r = R[j]
            Phi_xr.set_r(r)

            # Attack done here, and its recorded runtime starts and stops here
            t_in = time.perf_counter()
            d = attack(Phi_xr, Phi_xr.get_reference_for_attack(), df(yx), 0.5, lib=lib, algo=algo)
            d_scaled = Phi_xr.rescale_back_attack_direction(d)
            # d_scaled = d_scaled*r/torch.linalg.norm(d_scaled, ord=float("inf"))
            t_out = time.perf_counter()
            # End of attack and runtime record

            xd = x+d_scaled; oxd = obj(xd)
            success = (oxd > ox)
            dt = t_out-t_in

            S_x.append(success)
            T_x.append(dt)
            A_x.append(oxd)

            if verbose >= 3:
                str_trial   = str_digits.format(i*nb_r+j+1, nb_x*nb_r)
                str_runtime = "{:>9.4f}".format(dt*10**3)
                str_obj     = "{:>+11.4E}".format(ox)
                str_atk     = "{:>+11.4E}".format(oxd)
                str_delta   = "{:>+11.4E}".format(oxd-ox)
                str_radius  = "{:>1.0E}".format(r)
                str_success = "{:^7s}".format(str(success))
                str_summary = " | ".join(["{:s}"]*7).format(str_trial, str_runtime, str_obj, str_atk, str_delta, str_radius, str_success)
                print(str_summary)

        S.append(S_x)
        T.append(T_x)
        A.append(A_x)
        if verbose >= 3: print(str_sep)

    # Display per value of obj(x)
    if verbose >= 2:
        for i in range(nb_x):
            o = O[i]
            S_x = S[i]
            str_o = "{:+1.2E}".format(o)
            str_ratio = "{:>7.2%}".format(sum(S_x) / len(S_x))
            str_summary = "success ratio for x with f(Phi(x)) = {:s}: {:s}".format(str_o, str_ratio)
            print(str_summary)

    # Display per value of r
    if verbose >= 2:
        for j in range(nb_r):
            S_r = [Sx[j] for Sx in S]
            str_r = "{:1.0E}".format(R[j])
            str_ratio = "{:>7.2%}".format(sum(S_r) / len(S_r))
            str_summary = "success ratio for r = {:s}: {:s}".format(str_r, str_ratio)
            print(str_summary)

    # Global display
    if verbose >= 1:
        attack_runtime = torch.mean(torch.tensor(T)).item()
        ratio_success = sum([sum(S_x) for S_x in S]) / (nb_x*nb_r)
        str_runtime = "mean runtime = {:f} ms".format(attack_runtime*10**3)
        str_success = "mean success ratio = {:.2%}".format(ratio_success)
        str_summary = "{:s}, {:s}".format(str_runtime, str_success)
        print(str_summary); print()

    # Save all the data computed by this function
    history_analysis = [["X", "R", "obj(X)", "status of [[(x,r) for r in R] for x in X]", "runtime of [[(x,r) for r in R] for x in X]", "gain of [[(x,r) for r in R] for x in X]"],
                        [X, R, O, S, T, A],
                        ]
    name_to_save = "attack_analysis_("+lib+";"+algo+").pt"
    torch.save(history_analysis, "/".join([path_to_save, name_to_save]))
