#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#%% Libraries import

# Torch-related packages
import torch
from torch import nn



#%% Definition of Phi_(x,r)

class Phi_xr_model(nn.Module):

    def __init__(self, Phi, x, r, output_components_ignored=[]):
        super().__init__()
        self.Phi = Phi
        self.x = self.clone_torch_tensor(x)
        self.r = r
        self.o = torch.ones_like(x)/2
        self.output_components_ignored = output_components_ignored
        self.dr_to_xpd_layer = self.make_layer_dr_to_xpd()
        self.substract_layer = self.make_layer_substract_Phi_reference_output()
        self.remove_components_layer = self.make_layer_remove_output_components()

    def clone_torch_tensor(self, t): return t.clone().detach()

    def set_Phi(self, Phi):
        self.Phi = Phi
        self.substract_layer = self.make_layer_substract_Phi_reference_output()

    def set_x(self, x):
        self.x = self.clone_torch_tensor(x)
        self.dr_to_xpd_layer = self.make_layer_dr_to_xpd()
        self.substract_layer = self.make_layer_substract_Phi_reference_output()

    def set_r(self, r):
        self.r = r
        self.dr_to_xpd_layer = self.make_layer_dr_to_xpd()

    def set_o(self, o):
        self.o = o
        self.dr_to_xpd_layer = self.make_layer_dr_to_xpd()

    def make_layer_dr_to_xpd(self):
        n = self.Phi.n
        layer = nn.Linear(n, n)
        layer.weight = nn.Parameter(torch.eye(n)*2*self.r)
        layer.bias   = nn.Parameter(self.x - 2*self.r*self.o)
        return layer

    def make_layer_substract_Phi_reference_output(self):
        x   = self.x
        Phi = self.Phi
        m   = Phi.m
        layer = nn.Linear(m, m)
        layer.weight = nn.Parameter(torch.eye(m))
        layer.bias   = nn.Parameter(-1*Phi(x))
        return layer

    def make_layer_remove_output_components(self):
        m = self.Phi.m
        output_components_ignored = self.output_components_ignored
        m_removed = m-len(output_components_ignored)
        layer = nn.Linear(m, m_removed, bias=False)
        W = torch.zeros(m_removed, m)
        line = 0
        for dim in range(m):
            if not(dim in output_components_ignored):
                W[line,dim] = 1
                line += 1
        layer.weight = nn.Parameter(W)
        return layer

    def get_reference_for_attack(self): return self.o

    def rescale_back_attack_direction(self, d): return 2*self.r*(d-self.o)

    def forward(self, dr):
        xpd = self.dr_to_xpd_layer(dr)
        Phi_xpd = self.Phi(xpd)
        Phi_xpd_minus_Phi_x = self.substract_layer(Phi_xpd)
        output_removed_dims = self.remove_components_layer(Phi_xpd_minus_Phi_x)
        return output_removed_dims
