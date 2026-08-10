#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Torch-related packages
import torch

# Optimization algorithms
from src.optimization_algorithms.random_line_searches import optim_random_line_searches
from src.optimization_algorithms.local_attacks        import optim_local_attacks
from src.optimization_algorithms.direct_search_method import optim_direct_search_method
from src.optimization_algorithms.bfgs                 import optim_bfgs
from src.optimization_algorithms.backprop_ascent      import optim_backprop_ascent
from src.optimization_algorithms.hybrid_method        import optim_hybrid_method

# Function related to history of algorithms
from src.optimization_algorithms.tools.save_history import save_history
from src.plot_functions.tools.load_history import check_if_result_file_exists_and_is_complete



#%% Main function doing all the desired runs for a given (f, Phi) couple

def optimization_runner(f, df, Phi, x_0, parameters_dict, runs_dict, path_results_folder, seed, appendix_name_to_save="", global_verbosity=1, force_rerun=False):



    default_dict = {"do_run":    False, # To override with appropriate boolean
                    "name":      "optim_", # Add relevant name, eg. "+= <name>"
                    "verbose":   max(0, global_verbosity),
                    "r_0":       parameters_dict["r_0"],
                    "r_dsm_min": parameters_dict["r_dsm_min"],
                    "r_dsm_max": parameters_dict["r_dsm_max"],
                    "r_atk_min": parameters_dict["r_atk_min"],
                    "r_atk_max": parameters_dict["r_atk_max"],
                    "algo_atk":  "FFGSM",
                    "eval_max":  parameters_dict["eval_max"],
                    "t_max":     parameters_dict["t_max"],
                    "k_max":     parameters_dict["k_max"],
                    "search":    False,
                    "seed":      seed,
                    }



    # Steepest ascent algorithm
    dict_optim_backprop_ascent = default_dict.copy()
    dict_optim_backprop_ascent["do_run"]  = runs_dict["backprop"]
    dict_optim_backprop_ascent["name"]   += "backprop_ascent"

    name_to_save = dict_optim_backprop_ascent["name"]+appendix_name_to_save+".pt"
    path_to_save = "/".join([path_results_folder, name_to_save])
    if force_rerun or (dict_optim_backprop_ascent["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
        r_0           = dict_optim_backprop_ascent["r_0"]
        nb_points_max = dict_optim_backprop_ascent["eval_max"]
        runtime_max   = dict_optim_backprop_ascent["t_max"]
        k_max         = dict_optim_backprop_ascent["k_max"]
        search        = dict_optim_backprop_ascent["search"]
        verbose       = dict_optim_backprop_ascent["verbose"]
        seed          = dict_optim_backprop_ascent["seed"]
        history_backprop_ascent = optim_backprop_ascent(
            f, df, Phi, x_0, r_0,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            verbose_iterations = verbose,
            seed          = seed
            )
        save_history(history_backprop_ascent, path_to_save)


    # BFGS algorithm
    dict_optim_bfgs = default_dict.copy()
    dict_optim_bfgs["do_run"]  = runs_dict["bfgs"]
    dict_optim_bfgs["name"]   += "bfgs"

    name_to_save = dict_optim_bfgs["name"]+appendix_name_to_save+".pt"
    path_to_save = "/".join([path_results_folder, name_to_save])
    if force_rerun or (dict_optim_bfgs["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
        r_0           = dict_optim_bfgs["r_0"]
        nb_points_max = dict_optim_bfgs["eval_max"]
        runtime_max   = dict_optim_bfgs["t_max"]
        k_max         = dict_optim_bfgs["k_max"]
        search        = dict_optim_bfgs["search"]
        verbose       = dict_optim_bfgs["verbose"]
        seed          = dict_optim_bfgs["seed"]
        history_bfgs = optim_bfgs(
            f, df, Phi, x_0, r_0,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            verbose_iterations = verbose,
            seed          = seed
            )
        save_history(history_bfgs, path_to_save)



    # DFO algo based on line searches with radius rk and random directions
    dict_optim_random_line_searches = default_dict.copy()
    dict_optim_random_line_searches["do_run"]  = runs_dict["linesearch"]
    dict_optim_random_line_searches["name"]   += "line_searches"

    name_to_save = dict_optim_random_line_searches["name"]+appendix_name_to_save+".pt"
    path_to_save = "/".join([path_results_folder, name_to_save])
    if force_rerun or (dict_optim_random_line_searches["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
        r_0           = dict_optim_random_line_searches["r_0"]
        r_min         = dict_optim_random_line_searches["r_dsm_min"]
        r_max         = dict_optim_random_line_searches["r_dsm_max"]
        nb_points_max = dict_optim_random_line_searches["eval_max"]
        runtime_max   = dict_optim_random_line_searches["t_max"]
        k_max         = dict_optim_random_line_searches["k_max"]
        search        = dict_optim_random_line_searches["search"]
        verbose       = dict_optim_random_line_searches["verbose"]
        seed          = dict_optim_random_line_searches["seed"]
        history_random_line_searches = optim_random_line_searches(
            f, df, Phi, x_0, r_0,
            r_min         = r_min,
            r_max         = r_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            verbose_iterations = verbose,
            seed          = seed,
            )
        save_history(history_random_line_searches, path_to_save)



    # Usual CDSM from DFO
    dict_optim_direct_search_method = default_dict.copy()
    dict_optim_direct_search_method["do_run"]  = runs_dict["cdsm"]
    dict_optim_direct_search_method["name"]   += "dsm"

    name_to_save = dict_optim_direct_search_method["name"]+appendix_name_to_save+".pt"
    path_to_save = "/".join([path_results_folder, name_to_save])
    if force_rerun or (dict_optim_direct_search_method["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
        r_0           = dict_optim_direct_search_method["r_0"]
        r_min         = dict_optim_direct_search_method["r_dsm_min"]
        r_max         = dict_optim_direct_search_method["r_dsm_max"]
        nb_points_max = dict_optim_direct_search_method["eval_max"]
        runtime_max   = dict_optim_direct_search_method["t_max"]
        k_max         = dict_optim_direct_search_method["k_max"]
        search        = dict_optim_direct_search_method["search"]
        verbose       = dict_optim_direct_search_method["verbose"]
        seed          = dict_optim_direct_search_method["seed"]
        history_direct_search_method = optim_direct_search_method(
            f, df, Phi, x_0, r_0,
            r_min         = r_min,
            r_max         = r_max,
            nb_points_max = nb_points_max,
            runtime_max   = runtime_max,
            k_max         = k_max,
            enable_search = search,
            verbose_iterations = verbose,
            seed          = seed
            )
        save_history(history_direct_search_method, path_to_save)



    # Algo based only on local attacks of Phi in the direction df(Phi(x))
    attacks = ["FGSM", "FFGSM", "SimBA", "RFGSM", "PGD", "BIM"]
    for attack in attacks:

        dict_optim_local_attacks = default_dict.copy()
        dict_optim_local_attacks["do_run"]   = runs_dict["attacks" + "_" + attack]
        dict_optim_local_attacks["algo_atk"] = attack
        dict_optim_local_attacks["name"]    += "attacks(" + attack + ")"

        name_to_save = dict_optim_local_attacks["name"]+appendix_name_to_save+".pt"
        path_to_save = "/".join([path_results_folder, name_to_save])
        if force_rerun or (dict_optim_local_attacks["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
            r_0           = dict_optim_local_attacks["r_0"]
            r_min         = dict_optim_local_attacks["r_atk_min"]
            r_max         = dict_optim_local_attacks["r_atk_max"]
            nb_points_max = dict_optim_local_attacks["eval_max"]
            runtime_max   = dict_optim_local_attacks["t_max"]
            k_max         = dict_optim_local_attacks["k_max"]
            algo          = dict_optim_local_attacks["algo_atk"]
            search        = dict_optim_local_attacks["search"]
            verbose       = dict_optim_local_attacks["verbose"]
            seed          = dict_optim_local_attacks["seed"]
            history_local_attacks = optim_local_attacks(
                f, df, Phi, x_0, r_0,
                r_min         = r_min,
                r_max         = r_max,
                nb_points_max = nb_points_max,
                runtime_max   = runtime_max,
                k_max         = k_max,
                algo          = algo,
                enable_search = search,
                verbose_iterations = verbose,
                seed          = seed
                )
            save_history(history_local_attacks, path_to_save)



    # Hybrid Attack-cDSM algorithm
    for attack in attacks:

        dict_optim_hybrid_method = default_dict.copy()
        dict_optim_hybrid_method["do_run"]   = runs_dict["hybrid" + "_" + attack]
        dict_optim_hybrid_method["algo_atk"] = attack
        dict_optim_hybrid_method["name"]    += "hybrid(" + attack + ")"

        name_to_save = dict_optim_hybrid_method["name"]+appendix_name_to_save+".pt"
        path_to_save = "/".join([path_results_folder, name_to_save])
        if force_rerun or (dict_optim_hybrid_method["do_run"] and not check_if_result_file_exists_and_is_complete(path_results_folder, name_to_save)):
            r_0           = dict_optim_hybrid_method["r_0"]
            r_dsm_min     = dict_optim_hybrid_method["r_dsm_min"]
            r_dsm_max     = dict_optim_hybrid_method["r_dsm_max"]
            r_atk_min     = dict_optim_hybrid_method["r_atk_min"]
            r_atk_max     = dict_optim_hybrid_method["r_atk_max"]
            nb_points_max = dict_optim_hybrid_method["eval_max"]
            runtime_max   = dict_optim_hybrid_method["t_max"]
            k_max         = dict_optim_hybrid_method["k_max"]
            algo          = dict_optim_hybrid_method["algo_atk"]
            search        = dict_optim_hybrid_method["search"]
            verbose       = dict_optim_hybrid_method["verbose"]
            seed          = dict_optim_hybrid_method["seed"]
            history_hybrid_method = optim_hybrid_method(
                f, df, Phi, x_0, r_0,
                r_dsm_min     = r_dsm_min,
                r_dsm_max     = r_dsm_max,
                r_atk_min     = r_atk_min,
                r_atk_max     = r_atk_max,
                nb_points_max = nb_points_max,
                runtime_max   = runtime_max,
                k_max         = k_max,
                algo          = algo,
                enable_search = search,
                verbose_iterations = verbose,
                seed          = seed
                )
            save_history(history_hybrid_method, path_to_save)
