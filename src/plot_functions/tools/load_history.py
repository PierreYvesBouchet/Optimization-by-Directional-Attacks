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
