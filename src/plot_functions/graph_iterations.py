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
labels["brls"] = ["failure", "linesearch"]
labels["atck"] = ["failure", "attack", "search"]
labels["cdsm"] = ["failure", "poll", "search", "covering"]
labels["hybr"] = ["failure+failure", "failure+poll", "failure+search", "failure+covering", "attack+failure", "attack+poll", "attack+search", "attack+covering", "attack+skipped"]
labels["bfgs"] = ["failure", "linesearch"]

remove_search_from_labels = True
if remove_search_from_labels:
    # labels["brls"] not changed; the brls algo has no search step
    labels["atck"] = [s for s in labels["atck"] if "search" not in s]
    labels["cdsm"] = [s for s in labels["cdsm"] if "search" not in s]
    labels["hybr"] = [s for s in labels["hybr"] if "search" not in s]



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

    # Formula for placing the different algorithms on the y-axis
    def y_position_algos(i_algo, n_algos):
        return n_algos-1-i_algo

    # Formula for placing the different statuses on the y-axis
    def y_position(i_algo, i_status, n_statuses):
        return y_position_algos(i_algo, len(list_data_history)) + 0.05 + 0.95 * i_status / n_statuses

    # Plotting loop
    for i in range(len(list_data_history)):
        history, color, label = list_data_history[i]
        labels_algo = labels[map_label_to_simpler_label(label)]
        abscissas  = []
        ordinates  = []
        for iter_k in history[2:]: # Reject 0st element because = header and 1st because = initialization
            x, o, k, t, v, s = iter_k[:6]
            y = y_position(i, labels_algo.index(s), len(labels_algo))
            abscissas.append(k)
            ordinates.append(y)
        ax.plot(abscissas, ordinates, color=color, linewidth=1, alpha=0.5, label=label)
        ax.plot(abscissas, ordinates, color=color, linewidth=0, marker="o", markersize=2)
        for j in range(len(labels_algo)):
            y = y_position(i, j, len(labels_algo))
            ax.plot([0, K_max], [y, y], color="black", linewidth=0.5, alpha=0.5)
        ax.plot([0, K_max], [i,   i  ], color="black", linewidth=2)
        ax.plot([0, K_max], [i+1, i+1], color="black", linewidth=2)
        ax.text(-0.5, y_position_algos(i, len(list_data_history))+0.5, label, horizontalalignment="right", verticalalignment="center", rotation=90, fontsize=12)

    # Plots settings
    yticks_values = []
    yticks_labels = []
    for i in range(len(list_data_history)):
        label = list_data_history[i][2]
        labels_algo = labels[map_label_to_simpler_label(label)]
        # yticks_values.append(i)
        # yticks_labels.append("")
        for j in range(len(labels_algo)):
            yticks_values.append(y_position(i, j, len(labels_algo)))
            yticks_labels.append(labels_algo[j])
    # yticks_values.append(len(list_data_history))
    # yticks_labels.append("")
    ax.set_xticks(ticks=[i*10**(magnitude_K_max-1) for i in range(int(K_max/10**(magnitude_K_max-1))+1)]); ax.tick_params(axis="x", pad=0)
    ax.set_yticks(ticks=yticks_values, labels=yticks_labels); ax.yaxis.tick_right()
    ax.set_xlim(0, K_max)
    ax.set_ylim(0, len(list_data_history))
    ax.set_xlabel("iteration")
    # ax.set_ylabel("iterations status")
    fig.set_size_inches((10, 5))
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.07, top=0.99)
    # fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=len(list_data_history))

    fig.savefig("/".join([path_results_folder, "plot_iterations_results.pdf"]))
