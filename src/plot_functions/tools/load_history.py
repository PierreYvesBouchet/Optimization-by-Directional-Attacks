#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Function to load an history from path/file_name, if any

def load_history(file_name, path):
    path_file = "/".join([path, file_name])
    try:    history = torch.load(path_file, weights_only=True)
    except: history = None
    return history

def check_if_result_file_exists_and_is_complete(path_results_folder, name):
    path_to_save = "/".join([path_results_folder, name])
    try:
        history = load_history(name, path_results_folder)
        if history is None  : return False
        else                : return len(history) > 1
    except FileNotFoundError: return False
