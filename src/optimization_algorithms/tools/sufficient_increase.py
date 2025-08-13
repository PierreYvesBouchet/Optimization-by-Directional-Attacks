#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Sufficient increase function

def sufficient_increase(new_val, old_val, r, epsilon=1E-10, tau=0.5E-2):
    ratio = (new_val-old_val) / (abs(old_val)+epsilon)
    return ratio >= tau
