#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Torch-related packages
import torch

# Optimization algorithms
from src.optimization_algorithms.random_line_searches import optim_random_line_searches
from src.optimization_algorithms.local_attacks        import optim_local_attacks
from src.optimization_algorithms.direct_search_method import optim_direct_search_method
from src.optimization_algorithms.hybrid_method        import optim_hybrid_method

# Function to save history of algorithms
from src.optimization_algorithms.tools.save_history import save_history



#%% Main function doing all the desired runs for a given (f, Phi) couple

def optimization_runner(f, df, Phi, x_0, path_results_folder,
                        dict_optim_random_line_searches,
                        dict_optim_local_attacks,
                        dict_optim_direct_search_method,
                        dict_optim_hybrid_method,
                        ):

    # DFO algo based on line searches with radius rk and random directions
    name_to_save = dict_optim_random_line_searches["name"]+".pt"
    if dict_optim_random_line_searches["do_run"]:
        torch.manual_seed(0)
        r_0           = dict_optim_random_line_searches["r_0"]
        r_min         = dict_optim_random_line_searches["r_dsm_min"]
        r_max         = dict_optim_random_line_searches["r_dsm_max"]
        nb_points_max = dict_optim_random_line_searches["eval_max"]
        runtime_max   = dict_optim_random_line_searches["t_max"]
        k_max         = dict_optim_random_line_searches["k_max"]
        speculative   = dict_optim_random_line_searches["speculative"]
        verbose       = dict_optim_random_line_searches["verbose"]
        history_random_line_searches = optim_random_line_searches(
            f, df, Phi, x_0, r_0,
            r_min         = r_min,
            r_max         = r_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            enable_speculative_search = speculative,
            verbose_iterations = verbose,
            )
        path_to_save = "/".join([path_results_folder, name_to_save])
        save_history(history_random_line_searches, path_to_save)

    # Algo based only on local attacks of Phi in the direction df(Phi(x))
    name_to_save = dict_optim_local_attacks["name"]+".pt"
    if dict_optim_local_attacks["do_run"]:
        torch.manual_seed(0)
        r_0           = dict_optim_local_attacks["r_0"]
        r_min         = dict_optim_local_attacks["r_atk_min"]
        r_max         = dict_optim_local_attacks["r_atk_max"]
        nb_points_max = dict_optim_local_attacks["eval_max"]
        runtime_max   = dict_optim_local_attacks["t_max"]
        k_max         = dict_optim_local_attacks["k_max"]
        lib           = dict_optim_local_attacks["lib_atk"]
        algo          = dict_optim_local_attacks["algo_atk"]
        speculative   = dict_optim_local_attacks["speculative"]
        verbose       = dict_optim_local_attacks["verbose"]
        history_local_attacks = optim_local_attacks(
            f, df, Phi, x_0, r_0,
            r_min         = r_min,
            r_max         = r_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            lib           = lib,
            algo          = algo,
            enable_speculative_search = speculative,
            verbose_iterations = verbose,
            )
        path_to_save = "/".join([path_results_folder, name_to_save])
        save_history(history_local_attacks, path_to_save)

    # Usual DSM from DFO
    name_to_save = dict_optim_direct_search_method["name"]+".pt"
    if dict_optim_direct_search_method["do_run"]:
        torch.manual_seed(0)
        r_0           = dict_optim_direct_search_method["r_0"]
        r_min         = dict_optim_direct_search_method["r_dsm_min"]
        r_max         = dict_optim_direct_search_method["r_dsm_max"]
        nb_points_max = dict_optim_direct_search_method["eval_max"]
        runtime_max   = dict_optim_direct_search_method["t_max"]
        k_max         = dict_optim_direct_search_method["k_max"]
        speculative   = dict_optim_direct_search_method["speculative"]
        verbose       = dict_optim_direct_search_method["verbose"]
        history_direct_search_method = optim_direct_search_method(
            f, df, Phi, x_0, r_0,
            r_min         = r_min,
            r_max         = r_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            enable_speculative_search = speculative,
            verbose_iterations = verbose,
            )
        path_to_save = "/".join([path_results_folder, name_to_save])
        save_history(history_direct_search_method, path_to_save)

    # Hybrid Attack-cDSM algorithm
    name_to_save = dict_optim_hybrid_method["name"]+".pt"
    if dict_optim_hybrid_method["do_run"]:
        torch.manual_seed(0)
        r_0           = dict_optim_hybrid_method["r_0"]
        r_dsm_min     = dict_optim_hybrid_method["r_dsm_min"]
        r_dsm_max     = dict_optim_hybrid_method["r_dsm_max"]
        r_atk_min     = dict_optim_hybrid_method["r_atk_min"]
        r_atk_max     = dict_optim_hybrid_method["r_atk_max"]
        nb_points_max = dict_optim_hybrid_method["eval_max"]
        runtime_max   = dict_optim_hybrid_method["t_max"]
        k_max         = dict_optim_hybrid_method["k_max"]
        lib           = dict_optim_hybrid_method["lib_atk"]
        algo          = dict_optim_hybrid_method["algo_atk"]
        speculative   = dict_optim_hybrid_method["speculative"]
        verbose       = dict_optim_hybrid_method["verbose"]
        history_hybrid_method = optim_hybrid_method(
            f, df, Phi, x_0, r_0,
            r_dsm_min     = r_dsm_min,
            r_dsm_max     = r_dsm_max,
            r_atk_min     = r_atk_min,
            r_atk_max     = r_atk_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            lib           = lib,
            algo          = algo,
            enable_speculative_search = speculative,
            verbose_iterations = verbose,
            )
        path_to_save = "/".join([path_results_folder, name_to_save])
        save_history(history_hybrid_method, path_to_save)
