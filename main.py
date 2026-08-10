#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%% Libraries import

# General Python-related packages
import os
import numpy as np
import sys
import subprocess
import fcntl
import time
import random

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



#%% Script parameters collection

# Initialization of the parameters
problem_name                   = "" # must match a folder name in path_root/problems/
use_tilde_reformulation        = True # must remain to "True" since case "False" not properly implemented
run_optim_backprop_ascent      = False
run_optim_bfgs                 = False
run_optim_random_line_searches = False
run_optim_direct_search_method = False
run_optim_local_attacks_SimBA  = False
run_optim_local_attacks_FGSM   = False
run_optim_local_attacks_FFGSM  = False
run_optim_local_attacks_RFGSM  = False
run_optim_local_attacks_PGD    = False
run_optim_local_attacks_BIM    = False
run_optim_hybrid_method_SimBA  = False
run_optim_hybrid_method_FGSM   = False
run_optim_hybrid_method_FFGSM  = False
run_optim_hybrid_method_RFGSM  = False
run_optim_hybrid_method_PGD    = False
run_optim_hybrid_method_BIM    = False
do_plots                       = False
run_attack_analysis_backprop   = False
run_attack_analysis_SimBA      = False
run_attack_analysis_FGSM       = False
run_attack_analysis_FFGSM      = False
run_attack_analysis_RFGSM      = False
run_attack_analysis_PGD        = False
run_attack_analysis_BIM        = False
rebuild_problem                = False
nb_repeats                     = 5 # number of times to repeat the whole optimization process (for stochastic algos)
seed                           = 0 # seed for the random number generator (for stochastic algos) (each repeat increments the seed by 1)

# Block of code to collect this script's parameters from IDE or from terminal
is_run_from_ide = False
if is_run_from_ide:
    problem_name                   = "barycentric_image_into_resnet" # "bio_pinn" # "warcraft_map_counterfactual"
    run_optim_backprop_ascent      = False
    run_optim_bfgs                 = False
    run_optim_random_line_searches = False
    run_optim_direct_search_method = False
    run_optim_local_attacks_SimBA  = False
    run_optim_local_attacks_FGSM   = False
    run_optim_local_attacks_FFGSM  = False
    run_optim_local_attacks_RFGSM  = False
    run_optim_local_attacks_PGD    = False
    run_optim_local_attacks_BIM    = False
    run_optim_hybrid_method_SimBA  = False
    run_optim_hybrid_method_FGSM   = False
    run_optim_hybrid_method_FFGSM  = False
    run_optim_hybrid_method_RFGSM  = False
    run_optim_hybrid_method_PGD    = False
    run_optim_hybrid_method_BIM    = False
    run_attack_analysis_backprop   = False
    run_attack_analysis_SimBA      = False
    run_attack_analysis_FGSM       = False
    run_attack_analysis_FFGSM      = False
    run_attack_analysis_RFGSM      = False
    run_attack_analysis_PGD        = False
    run_attack_analysis_BIM        = False
    do_plots                       = True
    rebuild_problem                = False
    nb_repeats                     = 5
    seed                           = 0
else:
    problem_name = sys.argv[1]
    for arg in sys.argv[2:-2]:
        arg = int(arg)
        if arg ==  0: run_optim_backprop_ascent      = True
        if arg ==  1: run_optim_bfgs                 = True
        if arg ==  2: run_optim_random_line_searches = True
        if arg ==  3: run_optim_direct_search_method = True
        if arg ==  4: run_optim_local_attacks_SimBA  = True
        if arg ==  5: run_optim_local_attacks_FGSM   = True
        if arg ==  6: run_optim_local_attacks_FFGSM  = True
        if arg ==  7: run_optim_local_attacks_RFGSM  = True
        if arg ==  8: run_optim_local_attacks_PGD    = True
        if arg ==  9: run_optim_local_attacks_BIM    = True
        if arg == 10: run_optim_hybrid_method_SimBA  = True
        if arg == 11: run_optim_hybrid_method_FGSM   = True
        if arg == 12: run_optim_hybrid_method_FFGSM  = True
        if arg == 13: run_optim_hybrid_method_RFGSM  = True
        if arg == 14: run_optim_hybrid_method_PGD    = True
        if arg == 15: run_optim_hybrid_method_BIM    = True
        if arg == -1: do_plots                       = True
        if arg == -2: rebuild_problem                = True
        if arg == -3: run_attack_analysis_backprop   = True
        if arg == -4: run_attack_analysis_SimBA      = True
        if arg == -5: run_attack_analysis_FGSM       = True
        if arg == -6: run_attack_analysis_FFGSM      = True
        if arg == -7: run_attack_analysis_RFGSM      = True
        if arg == -8: run_attack_analysis_PGD        = True
        if arg == -9: run_attack_analysis_BIM        = True
    nb_repeats = int(sys.argv[-2])
    seed = int(sys.argv[-1])

# Force skipping all runs
force_runs_false = False



#%% Paths related to the problems

path_root = os.path.dirname(os.path.abspath(__file__))
path_folder_problem            = "/".join([path_root, "problems", problem_name])
path_folder_problem_definition = "/".join([path_folder_problem, "problem"])
path_folder_problem_results    = "/".join([path_folder_problem, "results"])
sys.path.append(path_folder_problem)
sys.path.append(path_folder_problem_definition)
sys.path.append(path_folder_problem_results)



#%% Function to load a JIT model with inter-process file locking

def load_model_safe(model_path, max_retries=100, delay_bounds=(2.0, 5.0)):
    delay = random.uniform(0, 5)
    time.sleep(delay)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    for attempt in range(max_retries):
        lock_file = None
        try:
            lock_file = open(model_path, "r+b")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            model = torch.jit.load(model_path)
            return model
        except (RuntimeError, OSError) as e:
            error_msg = str(e)
            if any(err_phrase in error_msg for err_phrase in ["PytorchStreamReader failed", "invalid header", "corrupted", "truncated"]):
                if attempt < max_retries - 1:
                    delay = random.uniform(delay_bounds[0], delay_bounds[1])
                    print(f"Attempt {attempt + 1}/{max_retries} failed with transient error: {error_msg}. Retrying in {delay:.2f} seconds...")
                    time.sleep(delay)
                else: raise RuntimeError(f"Failed to load model after {max_retries} attempts. Last error: {error_msg}")
            else: raise RuntimeError(f"Failed to load model due to non-transient error: {error_msg}")
        finally:
            if lock_file is not None:
                time.sleep(1)
                try: fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                except OSError: pass
                lock_file.close()

    raise RuntimeError(f"Failed to load model after {max_retries} attempts")



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
    from make import f_tilde as f, df_tilde as df, c
    Phi_tilde_path = "/".join([path_folder_problem_definition, "Phi_tilde.pt"])
    Phi = load_model_safe(Phi_tilde_path)
else:
    raise ValueError("Error: algorithm solving the non-reformulated problem not implemented. Set use_tilde_reformulation = True.")
    # from make import f, df, c
    # Phi_path = "/".join([path_folder_problem_definition, "Phi.pt"])
    # Phi = load_model_safe(Phi_path)

# Fixing the weights of the NN Phi
Phi.eval()
for param in Phi.parameters(): param.requires_grad = False



#%% Import of others problem parameters

# Imports the starting point
path_x_0 = "/".join([path_folder_problem_definition, "x_0.pt"])
x_0 = torch.load(path_x_0, weights_only=True)

# Imports the problem parameters file
path_parameters = "/".join([path_folder_problem_definition, "parameters.pt"])
parameters = torch.load(path_parameters, weights_only=True)

# Imports the global solution, if any
path_x_star = "/".join([path_folder_problem_definition, "x_star.pt"])
try:    (x_star, f_star) = torch.load(path_x_star, weights_only=True)
except: (x_star, f_star) = (None, None)
(x_star, f_star) = (None, None)

# Imports the lower and upper bounds on the obj function values, if any
path_f_bounds = "/".join([path_folder_problem_definition, "f_bounds.pt"])
try:    (f_min, f_max) = torch.load(path_f_bounds, weights_only=True)
except: (f_min, f_max) = (None, None)

# Parses the parameters from the file
r_0       = parameters[0] # starting radius for either the attack step and the dsm
r_atk_min = parameters[1] # minimal radius for attack step (if r_atk falls below, it is truncated)
r_atk_max = parameters[2] # maximal radius for attack step (if r_atk grows above, it is truncated)
r_dsm_min = parameters[3] # minimal radius for dsm step (if r_dsm falls below, it is truncated)
r_dsm_max = parameters[4] # maximal radius for dsm step (if r_dsm grows above, it is truncated)

# Values related to stopping criteria checked at the end of each iteration
# if problem_name == "bio_pinn": eval_max = 5000
# elif problem_name == "warcraft_map_counterfactual": eval_max = 2000
# elif problem_name == "barycentric_image_into_resnet": eval_max = 10000
eval_max = float("inf")
k_max = float("inf")
t_max = 6*60*60 # seconds
symlog_threshold = -1E-10 # Determines whether the plots should be generated in symlog scale, if positive
if problem_name == "bio_pinn":
    eval_max = float("inf")
    t_max = 300
    lim_values = (0.2, 0.65)
    symlog_threshold = -1
elif problem_name == "warcraft_map_counterfactual":
    eval_max = float("inf")
    t_max = 300
    lim_values = (-1e6, -1)
    symlog_threshold = 1E0
elif problem_name == "barycentric_image_into_resnet":
    eval_max = 30000
    t_max = 900
    lim_values = (-1, 0)
    symlog_threshold = 1E-9

parameters_dict = { "r_0"       : r_0,
                    "r_atk_min" : r_atk_min,
                    "r_atk_max" : r_atk_max,
                    "r_dsm_min" : r_dsm_min,
                    "r_dsm_max" : r_dsm_max,
                    "eval_max"  : eval_max,
                    "k_max"     : k_max,
                    "t_max"     : t_max}



#%% Path to save the results related to the problem (can be set manually)

# Creation of results_folder, if necessary
try_create_result_folder = True
if try_create_result_folder:
    make_whole_path = False # Hardcoded to False since it could be dangerous otherwise
    if make_whole_path: # Creates the whole path, if not already existing
        os.makedirs(path_folder_problem_results, exist_ok=False)
    else: # If only path_folder_results's leaf doesn't exist, it is created
        try:    os.mkdir(path_folder_problem_results)
        except: pass



#%% Preparation of all algorithms dicts of parameters (add lines if needed) and run of all algorithms

# Disables all runs if force_runs_false is set to True
if force_runs_false:
    run_optim_backprop_ascent      = False
    run_optim_bfgs                 = False
    run_optim_random_line_searches = False
    run_optim_direct_search_method = False
    run_optim_local_attacks_SimBA  = False
    run_optim_local_attacks_FGSM   = False
    run_optim_local_attacks_FFGSM  = False
    run_optim_local_attacks_RFGSM  = False
    run_optim_local_attacks_PGD    = False
    run_optim_local_attacks_BIM    = False
    run_optim_hybrid_method_SimBA  = False
    run_optim_hybrid_method_FGSM   = False
    run_optim_hybrid_method_FFGSM  = False
    run_optim_hybrid_method_RFGSM  = False
    run_optim_hybrid_method_PGD    = False
    run_optim_hybrid_method_BIM    = False
    run_attack_analysis_backprop   = False
    run_attack_analysis_SimBA      = False
    run_attack_analysis_FGSM       = False
    run_attack_analysis_FFGSM      = False
    run_attack_analysis_RFGSM      = False
    run_attack_analysis_PGD        = False
    run_attack_analysis_BIM        = False

# Aggregating runs-related booleans in a dict
run_dict = {
            "backprop"      : run_optim_backprop_ascent,
            "bfgs"          : run_optim_bfgs,
            "linesearch"    : run_optim_random_line_searches,
            "cdsm"          : run_optim_direct_search_method,
            "attacks_SimBA" : run_optim_local_attacks_SimBA,
            "attacks_FGSM"  : run_optim_local_attacks_FGSM,
            "attacks_FFGSM" : run_optim_local_attacks_FFGSM,
            "attacks_RFGSM" : run_optim_local_attacks_RFGSM,
            "attacks_PGD"   : run_optim_local_attacks_PGD,
            "attacks_BIM"   : run_optim_local_attacks_BIM,
            "hybrid_SimBA"  : run_optim_hybrid_method_SimBA,
            "hybrid_FGSM"   : run_optim_hybrid_method_FGSM,
            "hybrid_FFGSM"  : run_optim_hybrid_method_FFGSM,
            "hybrid_RFGSM"  : run_optim_hybrid_method_RFGSM,
            "hybrid_PGD"    : run_optim_hybrid_method_PGD,
            "hybrid_BIM"    : run_optim_hybrid_method_BIM,
            }

# Runs the optimization algorithms we asked to
if True in run_dict.values():
    for i in range(nb_repeats):
        print("run {:d}/{:d}".format(i+1, nb_repeats))
        optimization_runner(f, df, Phi, x_0, parameters_dict, run_dict, path_folder_problem_results, seed+i, appendix_name_to_save="_run"+str(seed+i), force_rerun=False)
        print("\n\n\n")



#%% Run of the attack analysis algorithm

algos_analysis = []
if run_attack_analysis_backprop: algos_analysis.append("backprop")
if run_attack_analysis_SimBA   : algos_analysis.append("SimBA")
if run_attack_analysis_FGSM    : algos_analysis.append("FGSM")
if run_attack_analysis_FFGSM   : algos_analysis.append("FFGSM")
if run_attack_analysis_RFGSM   : algos_analysis.append("RFGSM")
if run_attack_analysis_PGD     : algos_analysis.append("PGD")
if run_attack_analysis_BIM     : algos_analysis.append("BIM")

if len(algos_analysis) > 0:

    nb_pts_max = 101
    list_points = [h[0] for h in load_history("optim_dsm_run"+str(seed)+".pt", path_folder_problem_results)[1:]]
    nb_pts = len(list_points)
    if nb_pts > nb_pts_max: list_points = [list_points[int(i*nb_pts/nb_pts_max)] for i in range(nb_pts_max)] # Selects nb_pts_max points uniformly in the list

    exp_r_min = int(np.log10(r_atk_min))
    exp_r_max = int(np.log10(r_atk_max))+1

    for algo in algos_analysis: attack_analysis(f, df, Phi, list_points, path_folder_problem_results, algo=algo, exp_r_min=exp_r_min, exp_r_max=exp_r_max, verbose=3)



#%% Plots of all graphs, if asked to

if do_plots: plots_maker(path_folder_problem_results, symlog_threshold=symlog_threshold, seed=seed, nb_repeats=nb_repeats, lim_values=lim_values)



#%% Additional section for whatever post-run purpose

if is_run_from_ide:

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
        history_local_attacks_FGSM   = load_history("optim_attacks(FGSM)_run"   +str(seed+i)+".pt", path_folder_problem_results)
        history_local_attacks_FFGSM  = load_history("optim_attacks(FFGSM)_run"  +str(seed+i)+".pt", path_folder_problem_results)
        history_local_attacks_RFGSM  = load_history("optim_attacks(RFGSM)_run"  +str(seed+i)+".pt", path_folder_problem_results)
        history_local_attacks_PGD    = load_history("optim_attacks(PGD)_run"    +str(seed+i)+".pt", path_folder_problem_results)
        history_local_attacks_BIM    = load_history("optim_attacks(BIM)_run"    +str(seed+i)+".pt", path_folder_problem_results)
        history_local_attacks_SimBA  = load_history("optim_attacks(SimBA)_run"  +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_FGSM          = load_history("optim_hybrid(FGSM)_run"    +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_FFGSM         = load_history("optim_hybrid(FFGSM)_run"   +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_RFGSM         = load_history("optim_hybrid(RFGSM)_run"   +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_PGD           = load_history("optim_hybrid(PGD)_run"     +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_BIM           = load_history("optim_hybrid(BIM)_run"     +str(seed+i)+".pt", path_folder_problem_results)
        history_hybrid_SimBA         = load_history("optim_hybrid(SimBA)_run"   +str(seed+i)+".pt", path_folder_problem_results)
        history_direct_search        = load_history("optim_dsm_run"             +str(seed+i)+".pt", path_folder_problem_results)
        history_backprop_ascent      = load_history("optim_backprop_ascent_run" +str(seed+i)+".pt", path_folder_problem_results)
        history_bfgs                 = load_history("optim_bfgs_run"            +str(seed+i)+".pt", path_folder_problem_results)
        # history_random_line_searches = load_history("optim_line_searches_run"   +str(seed+i)+".pt", path_folder_problem_results)
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
