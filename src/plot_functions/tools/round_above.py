#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

import math



#%% Function rounding value to its first greater multiple of base

def round_above(value, base): return base*math.ceil(value/base)
