#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# General Python-related packages
import os
import numpy as np
import sys
import subprocess

# Torch-related packages
import torch

# Attack step efficiency analysis, script to run all optim algos, and code doing all results plots
from src.attack_analysis     import attack_analysis
from src.optimization_runner import optimization_runner
from src.plots_maker         import plots_maker

# Post-execution results loader for each algos
from src.plot_functions.tools.load_history import load_history

# MatPlotLib to close old active plots
import matplotlib.pyplot as plt
plt.close("all")

# Fixing seed for reproductibility (both Torch and SciPy rely on NumPy seed)
torch.manual_seed(0)



#%% Script parameters collection

# Initialization of the parameters
problem_name                   = "" # must match a folder name in path_root/problems/
use_tilde_reformulation        = True # must remain to "True" since case "False" not properly implemented
run_optim_hybrid_method        = False
run_optim_local_attacks        = False
run_optim_direct_search_method = False
run_optim_random_line_searches = False
do_plots                       = False
run_attack_analysis            = False
rebuild_problem                = False

# Block of code to collect this script's parameters from IDE or from terminal
is_run_from_ide = False
if is_run_from_ide:
    problem_name                   = "warcraft_map_counterfactual"
    run_optim_hybrid_method        = True
    run_optim_direct_search_method = True
    run_optim_local_attacks        = True
    run_optim_random_line_searches = True
    run_attack_analysis            = True
    do_plots                       = True
    rebuild_problem                = True
else:
    problem_name = sys.argv[1]
    for arg in sys.argv[2:]:
        arg = int(arg)
        if arg ==  0: run_optim_hybrid_method        = True
        if arg ==  1: run_optim_direct_search_method = True
        if arg ==  2: run_optim_local_attacks        = True
        if arg ==  3: run_optim_random_line_searches = True
        if arg == -1: do_plots                       = True
        if arg == -2: run_attack_analysis            = True
        if arg == -3: rebuild_problem                = True

# Paths related to the problems
path_root = os.path.dirname(os.path.abspath(__file__))
path_folder_problem            = "/".join([path_root, "problems", problem_name])
path_folder_problem_definition = "/".join([path_folder_problem,                "problem"])
path_folder_problem_results    = "/".join([path_folder_problem,                "results"])
sys.path.append(path_folder_problem)
sys.path.append(path_folder_problem_definition)
sys.path.append(path_folder_problem_results)



#%% Import of the functions f and df, and of the NN Phi

# Runs <path_folder_problem>/make.py to re-generate the problem from scratch
if rebuild_problem:
    path_folder_make_py = "/".join([path_folder_problem_definition, "make.py"])
    try:
        problem_maker = subprocess.run(["python", path_folder_make_py], capture_output=True, text=True)
        if problem_maker.returncode == 0:
            print("Problem {:s} generated successfully. Script maker outputs:".format(problem_name))
            print("\tstdout: {:s}".format(problem_maker.stdout))
            print("\tstderr: {:s}".format(problem_maker.stderr))
        else:
            print("Problem generation {:s} failed. Script maker outputs:".format(problem_name))
            print("\tstdout: {:s}".format(problem_maker.stdout))
            print("\tstderr: {:s}".format(problem_maker.stderr))
    except FileNotFoundError:
        print("Script maker not found.")

# Imports the goal functions f and f_tilde, and their gradients df and df_tilde, which must be defined within path_folder_problem/make.py
if use_tilde_reformulation:
    from make import f_tilde as f, df_tilde as df
    Phi_tilde_path = "/".join([path_folder_problem_definition, "Phi_tilde.pt"])
    Phi = torch.jit.load(Phi_tilde_path)
else:
    raise ValueError("Error: algorithm solving the non-refomulated problem not implemented. Set use_tilde_reformulation = True.")
    # from make import f, df, c
    # Phi_path = "/".join([path_folder_problem_definition, "Phi.pt"])
    # Phi = torch.jit.load(Phi_path)

# Fixing the weights of the NN Phi
Phi.eval()
for param in Phi.parameters(): param.requires_grad = False



#%% Import of others problem parameters

# Force skipping all runs
force_runs_false = False
if force_runs_false:
    run_optim_hybrid_method        = False
    run_optim_random_line_searches = False
    run_optim_local_attacks        = False
    run_optim_direct_search_method = False
    run_attack_analysis            = False

# Imports the starting point
path_x_0 = "/".join([path_folder_problem_definition, "x_0.pt"])
x_0 = torch.load(path_x_0, weights_only=True)

# Imports the problem parameters file
path_parameters = "/".join([path_folder_problem_definition, "parameters.pt"])
parameters = torch.load(path_parameters, weights_only=True)

# Imports the global solution, if any
path_x_star = "/".join([path_folder_problem_definition, "x_star.pt"])
try:    x_star = torch.load(path_x_star, weights_only=True)
except: x_star = None

# Parses the parameters from the file
r_0       = parameters[0] # starting radius for either the attack step and the dsm
r_atk_min = parameters[1] # minimal radius for attack step (if r_atk falls below, it is truncated)
r_atk_max = parameters[2] # maximal radius for attack step (if r_atk grows above, it is truncated)
r_dsm_min = parameters[3] # minimal radius for dsm step (if r_dsm falls below, it is truncated)
r_dsm_max = parameters[4] # maximal radius for dsm step (if r_dsm grows above, it is truncated)
global_verbosity = 1 # If in IN^*, overrides each optim algo's default verbosity

# Values related to stopping criteria checked at the end of each iteration
eval_max = 2E4 # alternatively, 1E3*(Phi.n+1) is a rule of thumb from DFO
k_max = float("inf")
t_max = 6*60*60 # seconds (currently, only runtime(evaluate_batch_directions) is recorded)



#%% Path to save the results related to the problem (can be set manually)

# Creation of results_folder, if necessary
try_create_result_folder = True
if try_create_result_folder:
    make_whole_path = False # Hardcoded to False since it could be dangerous otherwise
    if make_whole_path: # Creates the whole path, if not already existing
        os.makedirs(path_folder_problem_results, exist_ok=False)
    else: # If only path_folder_results's leaf doesn't exists, it is created
        try:    os.mkdir(path_folder_problem_results)
        except: pass



#%% Preparation of all algorithms dicts of parameters (add lines if needed)

default_dict = {"do_run":    False, # To override with appropriate boolean
                "name":      "optim_", # Add relevant name, eg. "+= <name>"
                "verbose":   max(0, global_verbosity),
                "r_0":       r_0,
                "r_dsm_min": r_dsm_min,
                "r_dsm_max": r_dsm_max,
                "r_atk_min": r_atk_min,
                "r_atk_max": r_atk_max,
                "lib_atk":   "default",
                "algo_atk":  "default",
                "eval_max":  eval_max,
                "t_max":     t_max,
                "k_max":     k_max,
                "speculative": False,
                }

# DFO algo based on line searches with radius r_k and random directions
dict_optim_random_line_searches = default_dict.copy()
dict_optim_random_line_searches["do_run"]  = run_optim_random_line_searches
dict_optim_random_line_searches["name"]   += "line_searches"

# Algo based only on local attacks of Phi in the direction df(Phi(x))
dict_optim_local_attacks = default_dict.copy()
dict_optim_local_attacks["do_run"]   = run_optim_local_attacks
dict_optim_local_attacks["algo_atk"] = "FGSM"
dict_optim_local_attacks["name"]    += "attacks"

# Usual cDSM from DFO
dict_optim_direct_search_method = default_dict.copy()
dict_optim_direct_search_method["do_run"]  = run_optim_direct_search_method
dict_optim_direct_search_method["name"]   += "dsm"

# Hybrid Attack-cDSM algorithm
dict_optim_hybrid_method = default_dict.copy()
dict_optim_hybrid_method["do_run"]   = run_optim_hybrid_method
dict_optim_hybrid_method["algo_atk"] = "FGSM"
dict_optim_hybrid_method["name"]    += "hybrid"



#%% Run of all optimization algorithms

optimization_runner(f, df, Phi, x_0, path_folder_problem_results,
                    dict_optim_random_line_searches,
                    dict_optim_local_attacks,
                    dict_optim_direct_search_method,
                    dict_optim_hybrid_method,
                    )



#%% Run of the attack analysis algorithm

# Chooses which set of points to consider in the experiment related to the attack operator
#   True  => those considered by the hybrid method during its optim (from <path_folder_results>/optim_hybrid.pt, if it exists, otherwise behaves as in case below)
#   False => those defined by <path_folder_problem>/attack_analysis_points.pt
take_points_from_hybrid = True

if run_attack_analysis:

    take_points_from_hybrid_overriden = False
    try:
        nb_pts_max = 200
        list_points = [h[0] for h in load_history("optim_hybrid.pt", path_folder_problem_results)[1:]]
        nb_pts = len(list_points)
        if nb_pts > nb_pts_max:
            list_points = [list_points[int(i*nb_pts/nb_pts_max)] for i in range(nb_pts_max)] # Selects nb_pts_max points uniformly in the list
    except:
        if take_points_from_hybrid: print("Attack analysis: import of points from hybrid method failed. Using default set of points instead.")
        take_points_from_hybrid_overriden = True

    if not(take_points_from_hybrid) or take_points_from_hybrid_overriden:
        path_list_points = "/".join([path_folder_problem_definition, "attack_analysis_points.pt"])
        list_points = torch.load(path_list_points, weights_only=True)

    exp_r_min = int(np.log10(r_atk_min))
    exp_r_max = int(np.log10(r_atk_max))+1

    for algo in ["FGSM", "PGD"]:
        attack_analysis(f, df, Phi, list_points, path_folder_problem_results,
                        algo=algo, exp_r_min=exp_r_min, exp_r_max=exp_r_max,
                        verbose=3)



#%% Plots of all graphs, if asked to

if do_plots: plots_maker(path_folder_problem_results)



#%% Additional section for whatever post-run purpose

if is_run_from_ide:

    try:    history_hybrid_method        = load_history("optim_hybrid.pt",        path_folder_problem_results)
    except: history_hybrid_method        = None

    try:    history_local_attacks        = load_history("optim_attacks.pt",       path_folder_problem_results)
    except: history_local_attacks        = None

    try:    history_direct_search_method = load_history("optim_dsm.pt",           path_folder_problem_results)
    except: history_direct_search_method = None

    try:    history_random_line_searches = load_history("optim_line_searches.pt", path_folder_problem_results)
    except: history_random_line_searches = None
