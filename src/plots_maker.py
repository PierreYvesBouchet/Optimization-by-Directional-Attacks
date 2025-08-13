#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Plot functions
from src.plot_functions.graph_attack_analysis import graph_attack_analysis
from src.plot_functions.graph_iterations      import graph_iterations
from src.plot_functions.graph_nb_pts          import graph_nb_pts
# from src.plot_functions.graph_run_time        import graph_run_time

# Function to load history of algorithms
from src.plot_functions.tools.load_history import load_history



#%% Main function doing all the desired runs for a given (f, Phi) couple

def plots_maker(path_results_folder):

    # Get records of the attack analyses for all implemented algo (for each algo, = None if no file)
    history_attack_FGSM = load_history("attack_analysis_(default;FGSM).pt", path_results_folder)
    history_attack_PGD  = load_history("attack_analysis_(default;PGD).pt",  path_results_folder)
    list_histories_attack = [[history_attack_FGSM, "FGSM", "red" ],
                             [history_attack_PGD,  "PGD",  "blue"],
                            ]

    # Get records of all implemented optim algo's histories (= None if none)
    history_random_line_searches = load_history("optim_line_searches.pt", path_results_folder)
    history_local_attacks        = load_history("optim_attacks.pt",       path_results_folder)
    history_direct_search_method = load_history("optim_dsm.pt",           path_results_folder)
    history_hybrid_method        = load_history("optim_hybrid.pt",        path_results_folder)
    list_histories_optim = [
                            [history_random_line_searches, "gold",  "$\mathbb{M}_{\mathrm{rls}}$"],
                            [history_local_attacks,        "green", "$\mathbb{M}_{\mathrm{atk}}$"],
                            [history_direct_search_method, "blue",  "$\mathbb{M}_{\mathrm{dsm}}$"],
                            [history_hybrid_method,        "red",   "$\mathbb{M}_{\mathrm{hyb}}$"],
                           ]

    graph_attack_analysis(path_results_folder, list_histories_attack)

    # graph_run_time(  path_results_folder, list_histories_optim)
    graph_nb_pts(    path_results_folder, list_histories_optim)
    graph_iterations(path_results_folder, list_histories_optim)
