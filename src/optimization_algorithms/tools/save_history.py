#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import torch



#%% Function to save the history of an algorithm's run

def save_history(history, path_to_save): torch.save(history, path_to_save)
