#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import os
import torch



#%% Function to import of the best solution known to date, if any

def load_best_solution(path_to_get):
    path_best_solution = "/".join([path_to_get, "best_solution.pt"])
    if os.path.isfile(path_best_solution):
        x_best = torch.load(path_best_solution, weights_only=True)
    else:
        x_best = None
    return x_best