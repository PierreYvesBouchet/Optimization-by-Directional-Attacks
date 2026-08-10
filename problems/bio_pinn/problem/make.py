# -*- coding: utf-8 -*-



#%% Libraries import

import math
import os
import sys
import matplotlib.pyplot as plt
plt.close("all")

import torch
from torch import nn

path_root = os.path.dirname(os.path.abspath(__file__))
path_build_data = "/".join([path_root, "build_data"])
sys.path.append(path_build_data)



#%% Choice of the device to support the NN (hardcoded to CPU for portability)

if   torch.cuda.is_available():         device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else:                                   device = "cpu"
device = "cpu"



#%% Import of the PINN

path_model = "/".join([path_build_data, "model.pt"])
pinn = torch.load(path_model, weights_only=False, map_location=torch.device(device)).eval()
for param in pinn.parameters(): param.requires_grad = False

# model input  = (t, Q): reaction time [s] and microwave power [W]
#                trained with t in [0, ~600] and Q in {4, 5, 6} W only
# model output = (TG, DG, MG, G, ME, T): glyceride concentrations [mol/L] and temperature [°C]
#                after a reaction of duration t under constant power Q, from fixed initial conditions.
# ME is the target biodiesel (methyl ester). TG -> DG+ME -> MG+ME -> G+ME (reversible steps).
# T must stay <= 65°C (above which methanol evaporates and the PINN is unreliable).



#%% Problem parameters

K = 40   # number of independent production batches
N = 10   # number of time steps evaluated per batch (indices 0,...,N inclusive, so N+1 points)

n = 2*K  # input dimension: x = (t_1, Q_1, t_2, Q_2, ..., t_K, Q_K)
m = 6*(N+1)*K  # Phi output dimension

weight_c = 1e1 # penalization weight for constraint violations in the tilde reformulation

# Per-batch operating bounds — equal for all batches, within the PINN training regime
t_max = 600.0  # [s]  upper bound on reaction time
Q_min =   0.0  # [W]  lower bound on microwave power
Q_max =  12.0  # [W]  upper bound on microwave power

# Sinusoidal energy price weights: w_k = 1 + alpha * sin(2*pi*k/K), k = 0,...,K-1.
# Prices span [1-alpha, 1+alpha] with exactly one full period across the K batches.
# Their sum equals K (the sinusoid averages to zero over a full period), so the mean
# price is 1 regardless of alpha, and E_total has a clean interpretation as K times
# an average per-batch energy budget.
alpha_energy   = 0.75
weights_energy = torch.tensor([1.0 + alpha_energy * math.sin(2.0*math.pi*k/K) for k in range(K)])

# Total weighted energy budget [J]: sum_k w_k * t_k * Q_k <= E_total.
# At equal allocation (all batches at t=60 s, Q=5 W): weighted total = K * 1 * 5 * 60 = 3000 J at K = 10.
# E_total is set to ~2/3 of the equal-allocation value so that the energy constraint is binding
# and the optimizer must genuinely differentiate across batches.
E_total = float(K) * 300.0



#%% Goal function: average final-time purity ratio across batches, and its gradient

# For batch k, the final-time purity ratio is:
#   purity_k = ME_k / (TG_k + DG_k + MG_k + G_k + ME_k), where all concentrations are evaluated at the final time step i=N.
# Objective: f(y) = (1/K) * sum_{k=0}^{K-1} purity_k
#
# Indexing convention in y (Phi output):
#   y[ 6*(k*(N+1)+i) + j ]   with j in {0:TG, 1:DG, 2:MG, 3:G, 4:ME, 5:T}
#   is the j-th output for batch k at time step i.
#
# Only the five concentration outputs at the final time step (i=N) of each batch appear in f.
# The temperature outputs and all intermediate time steps are in the inactive subspace of f.

def f(y, K:int=K, N:int=N):
    N_steps = N + 1
    total   = torch.zeros(1)
    for k in range(K):
        base  = 6*(k*N_steps + N)
        TG    = y[base]
        DG    = y[base+1]
        MG    = y[base+2]
        G     = y[base+3]
        ME    = y[base+4]
        total = total + ME / (TG + DG + MG + G + ME)
    return (total / K).item()

def df(y, K:int=K, N:int=N):
    N_steps = N + 1
    grad    = torch.zeros(y.shape)
    for k in range(K):
        base   = 6*(k*N_steps + N)
        TG     = y[base]
        DG     = y[base+1]
        MG     = y[base+2]
        G      = y[base+3]
        ME     = y[base+4]
        denom  = TG + DG + MG + G + ME
        denom2 = denom**2
        for j in range(4): grad[base+j] = -ME / denom2 / K # d/d(TG), d/d(DG), d/d(MG), d/d(G)
        grad[base+4] = (denom - ME) / denom2 / K           # d/d(ME)
        # grad[base+5] = 0                                 # d/d(T) — temperature absent from f
    return grad

# Components of y that do not influence f (inactive subspace of f):
# active indices are the 5 concentration outputs at the final time step of each batch.

_active_indices_f   = set(6*(k*(N+1)+N)+j for k in range(K) for j in range(5))
inactive_subspace_f = tuple(i for i in range(m) if i not in _active_indices_f)



#%% Constraints functions

# c_x: constraints depending only on x = (t_1, Q_1, ..., t_K, Q_K)
#   Box:    0 <= t_k <= t_max  and  Q_min <= Q_k <= Q_max  for all k  (4K constraints)
#   Energy: sum_k w_k * t_k * Q_k <= E_total                          ( 1 constraint)

def c_x(x,
        K:int=K,
        weights_energy:torch.Tensor=weights_energy,
        E_total:float=E_total,
        t_max:float=t_max,
        Q_min:float=Q_min,
        Q_max:float=Q_max):
    x_t        = x[0::2]   # reaction times  t_1, ..., t_K
    x_Q        = x[1::2]   # power levels    Q_1, ..., Q_K
    box_ctrs   = torch.cat([-x_t, x_t - t_max, Q_min - x_Q, x_Q - Q_max])
    energy_ctr = (weights_energy * x_t * x_Q).sum().unsqueeze(0) - E_total
    return torch.cat([box_ctrs, energy_ctr])

# c_y: constraints on the Phi output y (requires NN evaluation)
#   Non-negativity of all concentrations at every time step  (5*K*(N+1) constraints)
#   Temperature <= 65°C at every time step                   (  K*(N+1) constraints)
# Total: m = 6*K*(N+1) constraints.

def c_y(y, K:int=K, N:int=N):
    N_steps   = N + 1
    y_mat     = y.reshape(K*N_steps, 6)
    conc_ctrs = -y_mat[:, :5]           # -concentrations <= 0
    temp_ctrs =  y_mat[:, 5:] - 65.0    #  T - 65         <= 0
    return torch.cat([conc_ctrs.reshape(-1), temp_ctrs.reshape(-1)])

def c(x):
    y = Phi(x)
    return torch.cat([c_x(x), c_y(y)])



#%% Tilde reformulation: replace hard constraints by quadratic penalty on violations

# Phi_tilde(x) = [ Phi(x),  relu(c_x(x)),  relu(c_y(Phi(x))) ]
# f_tilde(yz)  = f(y) - weight_c * ||z||^2,   where yz = [y, z = relu(c)]

def f_tilde(yz, m:int=m, weight_c:float=weight_c):
    y = yz[:m]
    z = yz[m:]
    return (f(y) - weight_c * torch.linalg.norm(z)**2).item()

def df_tilde(yz, m:int=m, weight_c:float=weight_c):
    y = yz[:m]
    z = yz[m:]
    return torch.cat([df(y), -2.0*weight_c*z])



#%% Class generating the NN Phi

# Phi maps x = (t_1, Q_1, ..., t_K, Q_K) in R^{2K} to the stacked PINN predictions
# for all K batches, each evaluated at N+1 uniformly-spaced time fractions of t_k:
#   Phi(x) = [ pinn(i/N * t_k, Q_k) ]_{k=0,...,K-1 ; i=0,...,N}   in R^{6*K*(N+1)}
#
# The map from x to PINN inputs is linear and is implemented as a fixed-weight nn.Linear.
# For batch k and time step i:
#   pinn input row 2*(k*(N+1)+i)   = (i/N) * t_k   [column 2k   of weight matrix]
#   pinn input row 2*(k*(N+1)+i)+1 = Q_k           [column 2k+1 of weight matrix]

class Phi_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.K = K
        self.N = N
        self.n = n
        self.m = m
        self.inactive_subspace_f = list(inactive_subspace_f)  # List[int] for TorchScript compatibility
        self.pinn                = pinn
        self.rescale_x_layer         = self._make_rescale_x_layer()
        self.map_to_pinn_input_layer = self._make_pinn_input_layer()
        self.nb_forward_calls: int = 0

    def _make_rescale_x_layer(self) -> nn.Linear:
        layer = nn.Linear(self.n, self.n, bias=False)
        layer.weight = nn.Parameter(torch.eye(self.n))
        return layer

    def _make_pinn_input_layer(self) -> nn.Linear:
        W = torch.zeros(2*self.K*(self.N+1), self.n)
        for k in range(self.K):
            for i in range(self.N+1):
                W[2*(k*(self.N+1)+i),   2*k  ] = float(i) / float(self.N)
                W[2*(k*(self.N+1)+i)+1, 2*k+1] = 1.0
        layer = nn.Linear(self.n, 2*self.K*(self.N+1), bias=False)
        layer.weight = nn.Parameter(W)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.nb_forward_calls += 1
        x_rescaled      = self.rescale_x_layer(x)
        pinn_input_vec  = self.map_to_pinn_input_layer(x_rescaled)
        pinn_input      = pinn_input_vec.reshape(self.K*(self.N+1), 2)
        pinn_output     = self.pinn(pinn_input)
        pinn_output_vec = pinn_output.reshape(self.m, 1)
        if len(x.shape) == 1: pinn_output_vec = pinn_output_vec.squeeze(-1)
        return pinn_output_vec



#%% Class generating the NN Phi_tilde

class Phi_tilde_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.Phi  = Phi
        self.n    = self.Phi.n
        x0_probe  = torch.zeros(self.Phi.n)
        self.m    = self.Phi.m + len(c_x(x0_probe)) + len(c_y(Phi(x0_probe)))
        self.inactive_subspace_f = self.Phi.inactive_subspace_f
        self.relu = nn.ReLU()
        self.nb_forward_calls: int = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.nb_forward_calls += 1
        xs  = x.squeeze(0)
        y   = self.Phi(xs)
        z_x = self.relu(c_x(xs))
        z_y = self.relu(c_y(y))
        out = torch.cat([y, z_x, z_y], dim=-1)
        if len(x.shape) > 1:
            out = out.unsqueeze(0)
        return out



#%% Generation of the NNs Phi and Phi_tilde

Phi = Phi_model().to(device).eval()
for param in Phi.parameters(): param.requires_grad = False
Phi_scripted = torch.jit.script(Phi)
Phi_scripted.save("/".join([path_root, "Phi.pt"]))

Phi_tilde = Phi_tilde_model().to(device).eval()
for param in Phi_tilde.parameters(): param.requires_grad = False
Phi_tilde_scripted = torch.jit.script(Phi_tilde)
Phi_tilde_scripted.save("/".join([path_root, "Phi_tilde.pt"]))



#%% Problem data: starting point, radii, attack analysis points

# Starting point: all batches at (t=60 s, Q=5 W) — midpoint of the feasible box
x_0       = torch.zeros(n)
x_0[0::2] = 60.0   # t_k (s) for all k
x_0[1::2] =  2.0   # Q_k (W) for all k

# Step-size radii
r_0       = 1e0
r_atk_min = 1e-5;  r_atk_max = 2e1
r_dsm_min = 1e-5;  r_dsm_max = 2e1

# Lower and upper bound on the obj function values (used in plots only)
f_min = 0.2
f_max = 0.65

parameters = [r_0, r_atk_min, r_atk_max, r_dsm_min, r_dsm_max]

torch.save(x_0,                    "/".join([path_root, "x_0.pt"]))
torch.save(parameters,             "/".join([path_root, "parameters.pt"]))