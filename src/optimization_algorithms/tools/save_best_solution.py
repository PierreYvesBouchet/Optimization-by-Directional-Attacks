#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Function to save the input x_best as the best known solution

def save_best_solution(x_best, path_to_save): torch.save(x_best, path_to_save)
