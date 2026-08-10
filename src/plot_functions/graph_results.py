#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# MatPlotLib
import matplotlib.pyplot as plt

# Numpy for log10
import numpy as np

# Handmade function useful for plotting
from src.plot_functions.tools.round_above import round_above
from src.plot_functions.tools.convert_to_ms import convert_to_ms



#%% Graph of best objective value versus number of points evaluated

def graph_results(path_results_folder, list_data_history, symlog_y_threshold=0, lim_values=(-np.inf, np.inf), name_appendix=""):

    fig, ax = plt.subplots(ncols=2, sharey=True)

    # V_max = max(history_size(algo) for all algos)
    V_max = 0
    T_max = 0

    # First loop to get the values of V_max and T_max
    for histories, _, label, _ in list_data_history:
        for history in histories:
            V = 0
            T = 0
            try:
                for iter_k in history[1:]: # Reject 0st element because = header
                    x, o, k, t, v, s = iter_k[:6]
                    V += v
                    T += convert_to_ms(t)
            except Exception as e: print("Error while processing history {}: {}".format(label, e))
            V_max = max(V_max, V)
            T_max = max(T_max, T)
    magnitude_V_max = int(np.log10(V_max))
    magnitude_T_max = int(np.log10(T_max))
    V_max = round_above(V_max, 10**(magnitude_V_max-1))
    T_max = round_above(T_max, 10**magnitude_T_max)

    unit_converter = 1 if magnitude_T_max < 3 else 1e-3 # converts from ms to s if T_max > 10^5 ms = 100 s

    # Plots settings
    fig.subplots_adjust(left=0.10, right=0.98, bottom=0.13, top=0.85, wspace=0.04)
    fig.set_size_inches((8, 3))
    ax[0].set_xscale("symlog", linthresh=1e0); ax[1].set_xscale("symlog", linthresh=1e0)
    ax[0].tick_params(axis="x", pad=0); ax[1].tick_params(axis="x", pad=0)
    ax[0].set_xlim(0, V_max); ax[1].set_xlim(0, T_max * unit_converter)
    if lim_values is not None: ax[0].set_ylim(lim_values[0], lim_values[1]) ; ax[1].set_ylim(lim_values[0], lim_values[1])
    if symlog_y_threshold > 0: ax[0].set_yscale("symlog", linthresh=symlog_y_threshold); ax[1].set_yscale("symlog", linthresh=symlog_y_threshold)
    ax[0].set_xlabel("number of passes in $\\Phi$"); ax[1].set_xlabel("time " + ("[ms]" if unit_converter==1 else "[s]"))
    ax[0].set_ylabel("max value of $\\widetilde{f} \\circ \\widetilde{\\Phi}$ found")

    # Plotting loop
    for histories, color, label, ls in list_data_history:
        for i, history in enumerate(histories):
            V = 0
            T = 0
            abscissa_v = []
            abscissa_t = []
            ordinate = []
            try:
                for iter_k in history[1:]: # Reject 0st element because = header
                    x, o, k, t, v, s = iter_k[:6]
                    V += v
                    T += convert_to_ms(t) * unit_converter
                    abscissa_v.append(V)
                    abscissa_t.append(T)
                    ordinate.append(o)
            except Exception as e: print("Error while processing history {} for label {}: {}".format(i, label, e))
            ax[0].step(abscissa_v, ordinate, color=color, linewidth=2, alpha=0.5, linestyle=ls, where="post", label=(label if i==0 else "__nolegend__"))
            ax[1].step(abscissa_t, ordinate, color=color, linewidth=2, alpha=0.5, linestyle=ls, where="post")

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(list_data_history))
    fig.savefig("/".join([path_results_folder, "plot_results" + name_appendix + ".pdf"]))
