#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import



#%% # Function to unify the filling of algorithms history at each iteration
#   history = content of past iterations's outputs returned by fill_history
#   x = incumbent
#   o = f(Phi(x))
#   k = iteration number
#   t = total runtime of the evaluate() blocks at iteration k
#   v = total number of points evaluated at iteration k
#   s = comment on the iteration (eg. "failure", "sucess", "poll", "attack")
#   additional = array of potential algo-dependent information to store

def fill_history(history, x, o, k, t, v, s, additional = [], is_header=False):
    if is_header: history.append([x,                  o, k, t, v, s]+additional)
    else:         history.append([x.clone().detach(), o, k, t, v, s]+additional)
    return history
