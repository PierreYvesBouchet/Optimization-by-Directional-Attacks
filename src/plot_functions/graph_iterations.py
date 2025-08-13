#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# MatPlotLib
import matplotlib.pyplot as plt

# Numpy for log10
import numpy as np

# Handmade function useful for plotting
from src.plot_functions.tools.round_above import round_above



#%% Graph of best objective value versus number of points evaluated

# Expects as input labels of the form $\\mathbb{M}_{\\mathrm{name}}$
def map_label_to_simpler_label(label): return label.split("_")[1].replace("{\\mathrm{", "").replace("}}$", "")

# For each algorithm, list of all possible end-of-iteration statuses
labels = {}
labels["hyb"] = ["failure+failure", "failure+poll", "failure+search", "failure+covering", "attack+failure", "attack+poll", "attack+search", "attack+covering", "attack+skipped"]
labels["dsm"] = ["failure", "poll", "search", "covering"]
labels["atk"] = ["failure", "attack"]
labels["rls"] = ["failure", "linesearch"]



def graph_iterations(path_results_folder, list_data_history, K_max_upper_bound=1000):

    fig, ax = plt.subplots()

    # K_max = max(nb_iterations(algo) for all algos)
    K_max = 0

    # First loop to get K_max
    for history, _, _ in list_data_history:
        for iter_k in history[2:]: # Reject 0st element because = header and 1st because = initialization
            k = iter_k[2]
            K_max = max(K_max, k)
    K_max = min(K_max, K_max_upper_bound)
    magnitude_K_max = int(np.log10(K_max))
    K_max = round_above(K_max, 10**(magnitude_K_max-1))

    # Plotting loop
    for i in range(len(list_data_history)):
        history, color, label = list_data_history[i]
        labels_algo = labels[map_label_to_simpler_label(label)]
        abscissas  = []
        ordinates  = []
        for iter_k in history[2:]: # Reject 0st element because = header and 1st because = initialization
            x, o, k, t, v, s = iter_k[:6]
            y = i + 0.1 + 0.9 * labels_algo.index(s) / len(labels_algo)
            abscissas.append(k)
            ordinates.append(y)
        ax.plot(abscissas, ordinates, color=color, linewidth=1, alpha=0.5, label=label)
        ax.plot(abscissas, ordinates, color=color, linewidth=0, marker="o", markersize=4)
        for j in range(len(labels_algo)):
            y = i + 0.1 + 0.9 * j / len(labels_algo)
            ax.plot([0, K_max], [y, y], color="black", linewidth=0.5, alpha=0.5)
        ax.plot([0, K_max], [i,   i  ], color="black", linewidth=2)
        ax.plot([0, K_max], [i+1, i+1], color="black", linewidth=2)

    # Plots settings
    yticks_values = []
    yticks_labels = []
    for i in range(len(list_data_history)):
        label = list_data_history[i][2]
        labels_algo = labels[map_label_to_simpler_label(label)]
        yticks_values.append(i)
        yticks_labels.append("")
        for j in range(len(labels_algo)):
            yticks_values.append(i + 0.1 + 0.9 * j / len(labels_algo))
            yticks_labels.append(labels_algo[j])
    yticks_values.append(len(list_data_history))
    yticks_labels.append("")
    ax.set_yticks(ticks=yticks_values, labels=yticks_labels); ax.yaxis.tick_right()
    ax.set_xlim(0, K_max)
    ax.set_ylim(0, len(list_data_history))
    ax.set_xlabel("iteration")
    ax.set_ylabel("iterations status")
    fig.legend(loc="upper center", ncol=len(list_data_history))
    fig.set_size_inches((10, 5))
    fig.subplots_adjust(left=0.02, right=0.86, bottom=0.10, top=0.92)

    fig.savefig("/".join([path_results_folder, "plot_iterations_results.pdf"]))
