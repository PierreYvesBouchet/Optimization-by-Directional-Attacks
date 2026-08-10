#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Sufficient increase function

def sufficient_increase(new_val, old_val, r, epsilon=1E-10, tau=1E-2, force_simple_increase=False, force_false=False):
    if force_simple_increase: return new_val > old_val
    if force_false: return False
    return new_val >= old_val + tau*abs(old_val) + epsilon
