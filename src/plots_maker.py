#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Plot functions
from src.plot_functions.graph_attack_analysis import graph_attack_analysis
from src.plot_functions.graph_iterations      import graph_iterations
# from src.plot_functions.graph_nb_pts          import graph_nb_pts
# from src.plot_functions.graph_run_time        import graph_run_time
from src.plot_functions.graph_results          import graph_results

# Function to load history of algorithms
from src.plot_functions.tools.load_history import load_history



#%% Main function doing all the desired runs for a given (f, Phi) couple

def plots_maker(path_results_folder, symlog_threshold=1E-10, nb_repeats=1, theoretical_opt_value=None):

    # Get records of the attack analyses for all implemented algo (for each algo, = None if no file)
    history_attack_FGSM = load_history("attack_analysis_(default;FGSM).pt", path_results_folder)
    history_attack_PGD  = load_history("attack_analysis_(default;PGD).pt",  path_results_folder)
    list_histories_attack = [[history_attack_FGSM, "FGSM", "red" ],
                             [history_attack_PGD,  "PGD",  "blue"],
                            ]
    # Plot the attack analyses
    graph_attack_analysis(path_results_folder, list_histories_attack, symlog_y_threshold=symlog_threshold)

    # Get records of all implemented optim algo's first run histories (= None if none)
    history_random_line_searches = load_history("optim_line_searches_run0.pt", path_results_folder)
    history_local_attacks        = load_history("optim_attacks_run0.pt",       path_results_folder)
    history_direct_search_method = load_history("optim_dsm_run0.pt",           path_results_folder)
    history_hybrid_method        = load_history("optim_hybrid_run0.pt",        path_results_folder)
    history_bfgs_method          = load_history("optim_bfgs_run0.pt",          path_results_folder)
    list_histories_optim = [
                            [history_local_attacks,        "green", "$\mathbb{M}_{\mathrm{atck}}$"],
                            [history_bfgs_method,          "black", "$\mathbb{M}_{\mathrm{bfgs}}$"],
                            [history_random_line_searches, "gold",  "$\mathbb{M}_{\mathrm{brls}}$"],
                            [history_direct_search_method, "blue",  "$\mathbb{M}_{\mathrm{cdsm}}$"],
                            [history_hybrid_method,        "red",   "$\mathbb{M}_{\mathrm{hybr}}$"],
                           ]
    # Plot the results
    graph_iterations(path_results_folder, list_histories_optim)

    # Get records of all implemented optim algo's histories of all runs (= None if none)
    list_history_random_line_searches = []
    list_history_local_attacks        = []
    list_history_direct_search_method = []
    list_history_hybrid_method        = []
    list_history_bfgs_method          = []
    for i in range(nb_repeats):
        history_random_line_searches = load_history("optim_line_searches_run"+str(i)+".pt", path_results_folder)
        history_local_attacks        = load_history("optim_attacks_run"+str(i)+".pt",       path_results_folder)
        history_direct_search_method = load_history("optim_dsm_run"+str(i)+".pt",           path_results_folder)
        history_hybrid_method        = load_history("optim_hybrid_run"+str(i)+".pt",        path_results_folder)
        history_bfgs_method          = load_history("optim_bfgs_run"+str(i)+".pt",          path_results_folder)
        list_history_random_line_searches.append(history_random_line_searches)
        list_history_local_attacks.append(history_local_attacks)
        list_history_direct_search_method.append(history_direct_search_method)
        list_history_hybrid_method.append(history_hybrid_method)
        list_history_bfgs_method.append(history_bfgs_method)
    list_histories_optim = [
                            [list_history_local_attacks,        "green",  "$\mathbb{M}_{\mathrm{atck}}$"],
                            [list_history_bfgs_method,          "purple", "$\mathbb{M}_{\mathrm{bfgs}}$"],
                            [list_history_random_line_searches, "gold",   "$\mathbb{M}_{\mathrm{brls}}$"],
                            [list_history_direct_search_method, "blue",   "$\mathbb{M}_{\mathrm{cdsm}}$"],
                            [list_history_hybrid_method,        "red",    "$\mathbb{M}_{\mathrm{hybr}}$"],
                           ]
    graph_results(path_results_folder, list_histories_optim, symlog_y_threshold=symlog_threshold, theoretical_opt_value=theoretical_opt_value)
