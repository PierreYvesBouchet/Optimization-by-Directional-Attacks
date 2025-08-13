#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Convert the time t to milliseconds

def convert_to_ms(t, unit="s"):
    if unit == "s": return 10**3 * t
    if unit == "m": return convert_to_ms(60*t)
    if unit == "h": return convert_to_ms(60*t, unit="m")
    if unit == "d": return convert_to_ms(24*t, unit="h")
