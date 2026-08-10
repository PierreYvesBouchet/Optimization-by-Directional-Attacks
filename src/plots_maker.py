#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import numpy as np

# Plot functions
from src.plot_functions.graph_attack_analysis import graph_attack_analysis
from src.plot_functions.graph_iterations      import graph_iterations
# from src.plot_functions.graph_nb_pts          import graph_nb_pts
# from src.plot_functions.graph_run_time        import graph_run_time
from src.plot_functions.graph_results          import graph_results

# Function to load history of algorithms
from src.plot_functions.tools.load_history import load_history



#%% Pre-defined colors for the plots

# Color or attack algos (hybrid uses solid line, pure attack uses dashed line)
# Must exclude colors for others baselines gien below
color_FGSM    = "purple"
color_FFGSM   = "red"
color_RFGSM   = "darkred"
color_PGD     = "orange"
color_BIM     = "salmon"
color_SimBA   = "chocolate"

# Colors for other baselines (not attack-based)
color_backprop = "black"
color_dsm      = "blue"
color_bfgs     = "green"
# color_lsearch  = "gold"

# Legend for the plots:
legend_algos = {
    "attack_FGSM"   : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{fgsm}}$",
    "attack_FFGSM"  : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{ffgsm}}$",
    "attack_RFGSM"  : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{rfgsm}}$",
    "attack_PGD"    : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{pgd}}$",
    "attack_BIM"    : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{bim}}$",
    "attack_SimBA"  : r"$\mathbb{M}_{\mathrm{atck}}^{\mathrm{simba}}$",
    "hybrid_FGSM"   : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{fgsm}}$",
    "hybrid_FFGSM"  : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{ffgsm}}$",
    "hybrid_RFGSM"  : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{rfgsm}}$",
    "hybrid_PGD"    : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{pgd}}$",
    "hybrid_BIM"    : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{bim}}$",
    "hybrid_SimBA"  : r"$\mathbb{M}_{\mathrm{hybr}}^{\mathrm{simba}}$",
    "backprop"      : r"$\mathbb{M}_{\mathrm{back}}$",
    "cdsm"          : r"$\mathbb{M}_{\mathrm{cdsm}}$",
    "bfgs"          : r"$\mathbb{M}_{\mathrm{bfgs}}$",
    # "brls"          : r"$\mathbb{M}_{\mathrm{brls}}$",
}

# Linestyles for the plots
ls_attack = "--"
ls_hybrid = "-"
ls_backprop = "-"
ls_cdsm = "-"
ls_bfgs = "-"
# ls_brls = "-"




#%% Main function doing all the desired runs for a given (f, Phi) couple

def plots_maker(path_results_folder, symlog_threshold=1E-10, seed=0, nb_repeats=1, lim_values=(-np.inf, np.inf)):

    # Get records of the attack analyses for all implemented algo (for each algo, = None if no file)
    history_attack_FGSM = load_history("attack_analysis_FGSM.pt",       path_results_folder)
    history_attack_FFGSM= load_history("attack_analysis_FFGSM.pt",      path_results_folder)
    history_attack_RFGSM= load_history("attack_analysis_RFGSM.pt",      path_results_folder)
    history_attack_PGD  = load_history("attack_analysis_PGD.pt",        path_results_folder)
    history_attack_back = load_history("attack_analysis_backprop.pt",   path_results_folder)
    history_attack_SimBA= load_history("attack_analysis_SimBA.pt",      path_results_folder)
    history_attack_BIM  = load_history("attack_analysis_BIM.pt",        path_results_folder)
    list_histories_attack = [
                            #  [history_attack_FGSM, "FGSM",      color_FGSM      ],
                             [history_attack_PGD,  "PGD",       color_PGD       ],
                             [history_attack_FFGSM,"FFGSM",     color_FFGSM     ],
                             [history_attack_RFGSM,"RFGSM",     color_RFGSM     ],
                            #  [history_attack_BIM,  "BIM",       color_BIM       ],
                            #  [history_attack_SimBA,"SimBA",     color_SimBA     ],
                             [history_attack_back, "backprop",  color_backprop  ],
                            ]
    # Plot the attack analyses
    graph_attack_analysis(path_results_folder, list_histories_attack, symlog_y_threshold=symlog_threshold)

    # Get records of all implemented optim algo's first run histories (= None if none)
    history_hybrid_FGSM          = load_history("optim_hybrid(FGSM)_run"+str(seed)+".pt",    path_results_folder)
    history_hybrid_FFGSM         = load_history("optim_hybrid(FFGSM)_run"+str(seed)+".pt",   path_results_folder)
    history_hybrid_RFGSM         = load_history("optim_hybrid(RFGSM)_run"+str(seed)+".pt",   path_results_folder)
    history_hybrid_PGD           = load_history("optim_hybrid(PGD)_run"+str(seed)+".pt",     path_results_folder)
    history_hybrid_BIM           = load_history("optim_hybrid(BIM)_run"+str(seed)+".pt",     path_results_folder)
    history_hybrid_SimBA         = load_history("optim_hybrid(SimBA)_run"+str(seed)+".pt",   path_results_folder)
    history_local_attacks_FGSM   = load_history("optim_attacks(FGSM)_run"+str(seed)+".pt",   path_results_folder)
    history_local_attacks_FFGSM  = load_history("optim_attacks(FFGSM)_run"+str(seed)+".pt",  path_results_folder)
    history_local_attacks_RFGSM  = load_history("optim_attacks(RFGSM)_run"+str(seed)+".pt",  path_results_folder)
    history_local_attacks_PGD    = load_history("optim_attacks(PGD)_run"+str(seed)+".pt",    path_results_folder)
    history_local_attacks_BIM    = load_history("optim_attacks(BIM)_run"+str(seed)+".pt",    path_results_folder)
    history_local_attacks_SimBA  = load_history("optim_attacks(SimBA)_run"+str(seed)+".pt",  path_results_folder)
    history_direct_search        = load_history("optim_dsm_run"+str(seed)+".pt",             path_results_folder)
    history_backprop_ascent      = load_history("optim_backprop_ascent_run"+str(seed)+".pt", path_results_folder)
    history_bfgs                 = load_history("optim_bfgs_run"+str(seed)+".pt",            path_results_folder)
    # history_random_line_searches = load_history("optim_line_searches_run"+str(seed)+".pt",   path_results_folder)
    list_histories_optim = [
                            # [history_hybrid_FGSM,           color_FGSM,     legend_algos["hybrid_FGSM"],    ls_hybrid],
                            [history_hybrid_FFGSM,          color_FFGSM,    legend_algos["hybrid_FFGSM"],   ls_hybrid],
                            # [history_hybrid_RFGSM,          color_RFGSM,    legend_algos["hybrid_RFGSM"],   ls_hybrid],
                            # [history_hybrid_PGD,            color_PGD,      legend_algos["hybrid_PGD"],     ls_hybrid],
                            # [history_hybrid_BIM,            color_BIM,      legend_algos["hybrid_BIM"],     ls_hybrid],
                            # [history_hybrid_SimBA,          color_SimBA,    legend_algos["hybrid_SimBA"],   ls_hybrid],
                            # [history_local_attacks_FGSM,    color_FGSM,     legend_algos["attack_FGSM"],    ls_attack],
                            # [history_local_attacks_FFGSM,   color_FFGSM,    legend_algos["attack_FFGSM"],   ls_attack],
                            # [history_local_attacks_RFGSM,   color_RFGSM,    legend_algos["attack_RFGSM"],   ls_attack],
                            # [history_local_attacks_PGD,     color_PGD,      legend_algos["attack_PGD"],     ls_attack],
                            # [history_local_attacks_BIM,     color_BIM,      legend_algos["attack_BIM"],     ls_attack],
                            # [history_local_attacks_SimBA,   color_SimBA,    legend_algos["attack_SimBA"],   ls_attack],
                            [history_direct_search,         color_dsm,      legend_algos["cdsm"],           ls_cdsm],
                            # [history_bfgs,                  color_bfgs,     legend_algos["bfgs"],           ls_bfgs],
                            # [history_backprop_ascent,       color_backprop, legend_algos["backprop"],       ls_backprop],
                            # [history_random_line_searches,  color_lsearch, legend_algos["brls"],            ls_brls],
                           ]
    # Plot the results
    graph_iterations(path_results_folder, list_histories_optim)

    # Get records of all implemented optim algo's histories of all runs (= None if none)
    list_history_local_attacks_FGSM     = []
    list_history_local_attacks_FFGSM    = []
    list_history_local_attacks_RFGSM    = []
    list_history_local_attacks_PGD      = []
    list_history_local_attacks_BIM      = []
    list_history_local_attacks_SimBA    = []
    list_history_hybrid_FGSM            = []
    list_history_hybrid_FFGSM           = []
    list_history_hybrid_RFGSM           = []
    list_history_hybrid_PGD             = []
    list_history_hybrid_BIM             = []
    list_history_hybrid_SimBA           = []
    list_history_backprop_ascent        = []
    list_history_direct_search          = []
    list_history_bfgs                   = []
    # list_history_random_line_searches   = []
    for i in range(nb_repeats):
        history_local_attacks_FGSM   = load_history("optim_attacks(FGSM)_run"   +str(seed+i)+".pt", path_results_folder)
        history_local_attacks_FFGSM  = load_history("optim_attacks(FFGSM)_run"  +str(seed+i)+".pt", path_results_folder)
        history_local_attacks_RFGSM  = load_history("optim_attacks(RFGSM)_run"  +str(seed+i)+".pt", path_results_folder)
        history_local_attacks_PGD    = load_history("optim_attacks(PGD)_run"    +str(seed+i)+".pt", path_results_folder)
        history_local_attacks_BIM    = load_history("optim_attacks(BIM)_run"    +str(seed+i)+".pt", path_results_folder)
        history_local_attacks_SimBA  = load_history("optim_attacks(SimBA)_run"  +str(seed+i)+".pt", path_results_folder)
        history_hybrid_FGSM          = load_history("optim_hybrid(FGSM)_run"    +str(seed+i)+".pt", path_results_folder)
        history_hybrid_FFGSM         = load_history("optim_hybrid(FFGSM)_run"   +str(seed+i)+".pt", path_results_folder)
        history_hybrid_RFGSM         = load_history("optim_hybrid(RFGSM)_run"   +str(seed+i)+".pt", path_results_folder)
        history_hybrid_PGD           = load_history("optim_hybrid(PGD)_run"     +str(seed+i)+".pt", path_results_folder)
        history_hybrid_BIM           = load_history("optim_hybrid(BIM)_run"     +str(seed+i)+".pt", path_results_folder)
        history_hybrid_SimBA         = load_history("optim_hybrid(SimBA)_run"   +str(seed+i)+".pt", path_results_folder)
        history_direct_search        = load_history("optim_dsm_run"             +str(seed+i)+".pt", path_results_folder)
        history_backprop_ascent      = load_history("optim_backprop_ascent_run" +str(seed+i)+".pt", path_results_folder)
        history_bfgs                 = load_history("optim_bfgs_run"            +str(seed+i)+".pt", path_results_folder)
        # history_random_line_searches = load_history("optim_line_searches_run"   +str(seed+i)+".pt", path_results_folder)
        list_history_local_attacks_SimBA.append(    history_local_attacks_SimBA)
        list_history_local_attacks_FGSM.append(     history_local_attacks_FGSM)
        list_history_local_attacks_FFGSM.append(    history_local_attacks_FFGSM)
        list_history_local_attacks_RFGSM.append(    history_local_attacks_RFGSM)
        list_history_local_attacks_PGD.append(      history_local_attacks_PGD)
        list_history_local_attacks_BIM.append(      history_local_attacks_BIM)
        list_history_direct_search.append(          history_direct_search)
        list_history_hybrid_SimBA.append(           history_hybrid_SimBA)
        list_history_hybrid_FGSM.append(            history_hybrid_FGSM)
        list_history_hybrid_FFGSM.append(           history_hybrid_FFGSM)
        list_history_hybrid_RFGSM.append(           history_hybrid_RFGSM)
        list_history_hybrid_PGD.append(             history_hybrid_PGD)
        list_history_hybrid_BIM.append(             history_hybrid_BIM)
        list_history_backprop_ascent.append(        history_backprop_ascent)
        list_history_bfgs.append(                   history_bfgs)
        # list_history_random_line_searches.append(   history_random_line_searches)
    list_histories_optim = [
                            # [list_history_hybrid_FGSM,           color_FGSM,     legend_algos["hybrid_FGSM"],    ls_hybrid],
                            [list_history_hybrid_FFGSM,          color_FFGSM,    legend_algos["hybrid_FFGSM"],   ls_hybrid],
                            # [list_history_hybrid_RFGSM,          color_RFGSM,    legend_algos["hybrid_RFGSM"],   ls_hybrid],
                            # [list_history_hybrid_PGD,            color_PGD,      legend_algos["hybrid_PGD"],     ls_hybrid],
                            # [list_history_hybrid_BIM,            color_BIM,      legend_algos["hybrid_BIM"],     ls_hybrid],
                            # [list_history_hybrid_SimBA,          color_SimBA,    legend_algos["hybrid_SimBA"],   ls_hybrid],
                            # [list_history_local_attacks_FGSM,    color_FGSM,     legend_algos["attack_FGSM"],    ls_attack],
                            # [list_history_local_attacks_FFGSM,   color_FFGSM,    legend_algos["attack_FFGSM"],   ls_attack],
                            [list_history_local_attacks_RFGSM,   color_RFGSM,    legend_algos["attack_RFGSM"],   ls_attack],
                            # [list_history_local_attacks_PGD,     color_PGD,      legend_algos["attack_PGD"],     ls_attack],
                            # [list_history_local_attacks_BIM,     color_BIM,      legend_algos["attack_BIM"],     ls_attack],
                            # [list_history_local_attacks_SimBA,   color_SimBA,    legend_algos["attack_SimBA"],   ls_attack],
                            [list_history_direct_search,         color_dsm,      legend_algos["cdsm"],           ls_cdsm],
                            [list_history_bfgs,                  color_bfgs,     legend_algos["bfgs"],           ls_bfgs],
                            [list_history_backprop_ascent,       color_backprop, legend_algos["backprop"],       ls_backprop],
                            # [list_history_random_line_searches,   color_lsearch, legend_algos["brls"],            ls_brls],
                           ]
    graph_results(path_results_folder, list_histories_optim, symlog_y_threshold=symlog_threshold, lim_values=lim_values)

    # Same comparing only pure attack-based methods
    list_histories_optim = [
                            [list_history_local_attacks_FGSM,    color_FGSM,     legend_algos["attack_FGSM"],    ls_attack],
                            [list_history_local_attacks_FFGSM,   color_FFGSM,    legend_algos["attack_FFGSM"],   ls_attack],
                            [list_history_local_attacks_RFGSM,   color_RFGSM,    legend_algos["attack_RFGSM"],   ls_attack],
                            [list_history_local_attacks_PGD,     color_PGD,      legend_algos["attack_PGD"],     ls_attack],
                            [list_history_local_attacks_BIM,     color_BIM,      legend_algos["attack_BIM"],     ls_attack],
                            [list_history_local_attacks_SimBA,   color_SimBA,    legend_algos["attack_SimBA"],   ls_attack],
                            # [list_history_backprop_ascent,       color_backprop, legend_algos["backprop"],       ls_backprop],
                           ]
    graph_results(path_results_folder, list_histories_optim, symlog_y_threshold=symlog_threshold, lim_values=lim_values, name_appendix="_attacks_comparison")

    # Same comparing hybrid methods with various attacks
    list_histories_optim = [
                            [list_history_hybrid_FGSM,           color_FGSM,     legend_algos["hybrid_FGSM"],    ls_hybrid],
                            [list_history_hybrid_FFGSM,          color_FFGSM,    legend_algos["hybrid_FFGSM"],   ls_hybrid],
                            [list_history_hybrid_RFGSM,          color_RFGSM,    legend_algos["hybrid_RFGSM"],   ls_hybrid],
                            [list_history_hybrid_PGD,            color_PGD,      legend_algos["hybrid_PGD"],     ls_hybrid],
                            [list_history_hybrid_BIM,            color_BIM,      legend_algos["hybrid_BIM"],     ls_hybrid],
                            [list_history_hybrid_SimBA,          color_SimBA,    legend_algos["hybrid_SimBA"],   ls_hybrid],
                            # [list_history_backprop_ascent,       color_backprop, legend_algos["backprop"],       ls_backprop],
                           ]
    graph_results(path_results_folder, list_histories_optim, symlog_y_threshold=symlog_threshold, lim_values=lim_values, name_appendix="_hybrid_comparison")
