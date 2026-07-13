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

def graph_results(path_results_folder, list_data_history, scale_plots_to_best=False, symlog_y_threshold=0, theoretical_opt_value=None):

    fig, ax = plt.subplots(ncols=2, sharey=True)

    # V_max = max(history_size(algo) for all algos)
    # o_min = min(f(Phi(x)) for all x in history(algo) for all algos)
    # o_max = max(f(Phi(x)) for all x in history(algo) for all algos)
    V_max = 0
    T_max = 0
    o_min = +float("inf")
    o_max = -float("inf")

    # First loop to get the values of o_min, o_max and V_max
    for histories, _, _ in list_data_history:
        for history in histories:
            V = 0
            T = 0
            for iter_k in history[1:]: # Reject 0st element because = header
                x, o, k, t, v, s = iter_k[:6]
                V += v
                T += convert_to_ms(t)
                o_min = min(o_min, o)
                o_max = max(o_max, o)
            V_max = max(V_max, V)
            T_max = max(T_max, T)
    magnitude_V_max = int(np.log10(V_max))
    magnitude_T_max = int(np.log10(T_max))
    V_max = round_above(V_max, 10**(magnitude_V_max-1))
    T_max = round_above(T_max, 10**magnitude_T_max)

    unit_converter = 1 if magnitude_T_max < 3 else 1e-3 # converts from ms to s if T_max > 10^5 ms = 100 s


    # Plotting loop
    for histories, color, label in list_data_history:
        for i, history in enumerate(histories):
            V = 0
            T = 0
            abscissa_v = []
            abscissa_t = []
            ordinate = []
            for iter_k in history[1:]: # Reject 0st element because = header
                x, o, k, t, v, s = iter_k[:6]
                V += v
                T += convert_to_ms(t) * unit_converter
                abscissa_v.append(V)
                abscissa_t.append(T)
                o_plot = (o-o_max if scale_plots_to_best else o)
                ordinate.append(o_plot)
            ax[0].plot(abscissa_v, ordinate, color=color, linewidth=2, alpha=0.5, label=(label if i==0 else "__nolegend__"))
            ax[1].plot(abscissa_t, ordinate, color=color, linewidth=2, alpha=0.5)
        if theoretical_opt_value is not None:
            ax[0].axhline(y=theoretical_opt_value, color="black", linewidth=1, alpha=0.5, linestyle="--")
            ax[1].axhline(y=theoretical_opt_value, color="black", linewidth=1, alpha=0.5, linestyle="--")

    # Plots settings
    ax[0].set_xlim(0, V_max)
    ax[1].set_xlim(0, T_max * unit_converter)
    if scale_plots_to_best:
        ax[0].set_ylim(o_min-o_max, 0);
        ax[1].set_ylim(o_min-o_max, 0)
    else:
        # ymin = o_min-0.05*(o_max-o_min)
        # ymax = o_max+0.05*(o_max-o_min) if theoretical_opt_value is None else theoretical_opt_value+0.05*(o_max-theoretical_opt_value)
        ymin = o_min
        ymax = o_max if theoretical_opt_value is None else theoretical_opt_value
        ax[0].set_ylim(ymin, ymax)
        ax[1].set_ylim(ymin, ymax)
    if symlog_y_threshold > 0: ax[0].set_yscale("symlog", linthresh=symlog_y_threshold); ax[1].set_yscale("symlog", linthresh=symlog_y_threshold)
    ax[0].set_xlabel("number of trial points evaluated")
    ax[0].set_ylabel("best objective value found")
    ax[1].set_xlabel("time " + ("[ms]" if unit_converter==1 else "[s]"))
    fig.legend(loc="upper center", ncol=len(list_data_history))
    # ax.set_xticks(ticks=[i*10**(magnitude_V_max-1) for i in range(int(V_max/10**(magnitude_V_max-1))+1)]);
    ax[0].tick_params(axis="x", pad=0)
    ax[1].tick_params(axis="x", pad=0)
    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.12, top=0.88, wspace=0.05)
    if scale_plots_to_best: fig.subplots_adjust(left=0.11)
    fig.set_size_inches((8, 3))

    fig.savefig("/".join([path_results_folder, "plot_results.pdf"]))
