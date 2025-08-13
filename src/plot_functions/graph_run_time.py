#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# MatPlotLib
import matplotlib.pyplot as plt

# Numpy for log10
import numpy as np

# Handmade function useful for plotting
from src.plot_functions.tools.round_above   import round_above
from src.plot_functions.tools.convert_to_ms import convert_to_ms



#%% Graph of best objective value versus total runtime in the "evaluate" block

def graph_run_time(path_results_folder, list_data_history, scale_plots_to_best=False):

    fig, ax = plt.subplots()

    # T_max = max(runtime(algo) for all algos)
    # o_min = min(f(Phi(x)) for all x in history(algo) for all algos)
    # o_max = max(f(Phi(x)) for all x in history(algo) for all algos)
    T_max = 0
    o_min = +float("inf")
    o_max = -float("inf")

    # First loop to get the values of o_min, o_max and T_max
    for history, _, _ in list_data_history:
        T = 0
        for iter_k in history[1:]: # Reject 0st element because = header
            x, o, k, t, v, s = iter_k[:6]
            T += convert_to_ms(t)
            o_min = min(o_min, o)
            o_max = max(o_max, o)
        T_max = max(T_max, T)
    magnitude_T_max = int(np.log10(T_max))
    T_max = round_above(T_max, 10**magnitude_T_max)

    # Plotting loop
    for history, color, label in list_data_history:
        T = 0
        abscissa = []
        ordinate = []
        for iter_k in history[1:]: # Reject 0st element because = header
            x, o, k, t, v, s = iter_k[:6]
            T += convert_to_ms(t)
            abscissa.append(T)
            o_plot = (o-o_max if scale_plots_to_best else o)
            ordinate.append(o_plot)
        ax.plot(abscissa, ordinate, color=color, label=label, linewidth=2)

    # Plots settings
    ax.set_xlim(0, T_max)
    ax.set_xscale("symlog")
    if scale_plots_to_best:
        ax.set_ylim(o_min-o_max, 0)
        ax.set_yscale("symlog", linthresh=1E-10)
    else:
        ax.set_ylim(o_min, o_max)
    ax.set_xlabel("runtime (ms)")
    ax.set_ylabel("highest objective value found within the trial points")
    # fig.legend(loc="upper center", ncol=len(list_data_history))
    ax.legend(loc="lower right")
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.15, top=0.98)
    if scale_plots_to_best:
        fig.subplots_adjust(left=0.11)
    fig.set_size_inches((8, 3))

    fig.savefig("/".join([path_results_folder, "plot_run_time.pdf"]))
