#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# MatPlotLib, homemade plotting functions, and force closing all active figures
import matplotlib.pyplot as plt

# Numpy
import numpy as np

# Handmade function useful for plotting
from src.plot_functions.tools.round_above import round_above



#%% Best objective value versus number of points evaluated

def graph_attack_analysis(folder_path, list_data_history, nb_x_max=float("inf"), yscale_symlog=False):



    fig, axs = plt.subplots(nrows=1+len(list_data_history), sharex=True)
    fig.subplots_adjust(hspace=0.05)



    # First axis: overall info at given point x
        # abscissa = index i of the point in the list
        # left ordinate = f(Phi(x_i))
        # right ordinate = runtime of attack algos at x_i

    ax1 = axs[0]
    ax2 = ax1.twinx()

    for i in range(len(list_data_history)):

        history, label, color = list_data_history[i]
        X, R, O, S, T, A = history[1]
        nb_x = min(len(X), nb_x_max)
        nb_r = len(R)

        ax1.plot([i_x for i_x in range(nb_x)], [O[i_x] for i_x in range(nb_x)], color="black")
        # for i_x in range(nb_x):
        #     ax1.plot([0, i_x], [O[i_x], O[i_x]], color="black", alpha=0.25, lw=0.25, label="_nolegend_")

        for i_r in range(nb_r):
            ax2.plot([i_x for i_x in range(nb_x)], [T[i_x][i_r] for i_x in range(nb_x)], label=(label if i_r == 0 else "_nolegend_"), alpha=0.1, color=color)

    o_min = np.min(O)
    o_max = np.max(O)
    ax1.set_xlim(-0.5, nb_x-1+0.5)
    ax1.set_ylim(o_min, o_max)
    if yscale_symlog: ax1.set_yscale("symlog", linthresh=1E-10)
    ax1.set_ylabel("$f(\\Phi(x^j))$")

    t_max = np.max(T)
    ax2.set_ylim(0, round_above(t_max, 0.1))
    ax2.set_ylabel("t [s]")


    # Second and third axes: one axis per attack algo
        # abscissa = index i of the point in the list
        # ordinate = radius r of attack

    for i in range(len(list_data_history)):
        ax = axs[i+1]
        history, label, color = list_data_history[i]
        nb_x = 1

        if not(history == []):
            X, R, O, S, T, A = history[1]
            nb_x = min(len(X), nb_x_max)
            nb_r = len(R)

            for i_r in range(nb_r):
                for i_x in range(nb_x):
                    s = S[i_x][i_r]
                    if s: marker = "o"; alpha = 1.0; color_success = color
                    else: marker = "x"; alpha = 0.5; color_success = "dark"+color
                    ax.plot(i_x, i_r, marker=marker, color=color_success, markersize=3, alpha=alpha)

            ax.set_ylabel("r")
            ax.set_xlim(-0.5, nb_x-1+0.5)
            ax.set_ylim(-0.5, nb_r-1+0.5)

            # if   nb_x <= 100: xticks_values = [i for i in range(nb_x)]
            # elif nb_x <= 200: xticks_values = [i for i in range(0, nb_x, 5)]
            # else:             xticks_values = [int(i*nb_x/100) for i in range(100)]
            # xticks_minor_values = [i for i in range(nb_x)]
            # ax.set_xticks(ticks=xticks_values); ax.set_xticks(ticks=xticks_minor_values, minor=True); ax.tick_params(axis='x', labelrotation=90)
            # ax.set_yticks(ticks=[i for i in range(nb_r)], labels=["{:1.0E}".format(r) for r in R])

            xticks_values = [i for i in range(0, nb_x, 10)]
            xticks_minor_values = [i for i in range(nb_x)]
            ax.set_xticks(ticks=xticks_values); ax.set_xticks(ticks=xticks_minor_values, minor=True)
            ax.set_yticks(ticks=[i for i in range(nb_r)], labels=["{:1.0E}".format(r) for r in R])

            ax.text(nb_x-1+1.5, (nb_r-1+0.5)/2, label, rotation=90, ha='left', va='center')
            # ax.set_title(label, rotation=-90, position=(1, 0), ha='left', va='center')

    axs[-1].set_xlabel("Point index $j$")

    width = 12#max(12, 12*nb_x/100)
    fig.set_size_inches((width, 2*(1+len(list_data_history))+0.5))
    fig.subplots_adjust(left=0.07, right=0.95, bottom=0.07, top=0.99, hspace=0.05)

    fig.savefig("/".join([folder_path, "plot_attack_analysis.pdf"]))
