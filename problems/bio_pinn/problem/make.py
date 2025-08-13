# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import os
import sys
import matplotlib.pyplot as plt
plt.close("all")

# Torch-related packages
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

# PINN import in eval mode
path_model = "/".join([path_build_data, "model.pt"])
pinn = torch.load(path_model, weights_only=False, map_location=torch.device(device)).eval()
for param in pinn.parameters(): param.requires_grad = False

# model input  = tuple of (t, Q) = (time, microwave power), trained with 0 <= t <= 60 and Q in {4,5,6} only (otherwise may raise negative concentrations)
# model output = tuple of (concentration TG, concentration DG, concentration MG, concentration G, concentration ME, T) if warmed by Q during time t (from fixed initial conditions)
# the input may also be a tensor with all elements having the "input format" above, then the ouput is a tensor with all elements having the "output format" above
# ME is the biodiesel that we eventually want to maximize
# TG, DG, MG and G are the reactant we transform into ME: TG --> DG+ME and DG --> MG+ME and MG --> G+ME (and all reactions are reversible)
# T must remain <= 65 (ME changes phase then T > 65 so the PINN may be wrong)



#%% Problem definition, goal function and constraints function

# (n,m) = (Phi inputs dimension, Phi outputs dimension)
n = 2
N = 100
m = 6*N

# Normalization of x into [0,1]^2, and if so, coefficients to rescale x to its normal scale
# The rescaling coeffs have to bet set manually w.r.t. the c_x constraint, see below
normalize_x = True


# Goal function to be maximized:
#   f(y)   =   1/N  *  sum_{ti = t*i/N, i in range(1,N+1)} ME(ti) / (TG(ti)+DG(ti)+MG(ti))
def  f(y):
    N = int(len(y)/6)
    return sum([y[6*i+4] / (y[6*i]+y[6*i+1]+y[6*i+2]) for i in range(N)]).item()/N

def df(y):
    output = torch.zeros(y.shape)
    N = int(len(y)/6)
    for i in range(N):
        output[6*i  ] = -1*y[6*i+4] / (y[6*i]+y[6*i+1]+y[6*i+2])**2
        output[6*i+1] = -1*y[6*i+4] / (y[6*i]+y[6*i+1]+y[6*i+2])**2
        output[6*i+2] = -1*y[6*i+4] / (y[6*i]+y[6*i+1]+y[6*i+2])**2
        output[6*i+3] =  0
        output[6*i+4] =           1 / (y[6*i]+y[6*i+1]+y[6*i+2])
        output[6*i+5] =  0
    return output/N

# Components of the vector y that do not influence f
inactive_subspace_f = tuple([6*i+3 for i in range(N)]+[6*i+5 for i in range(N)])


# Constraints
#    0 <= t <= 120
#    0 <= Q <= 20
#   Qt <= 500(Q*t = energy consumption [w.s = J])
#    0 <= concentrations
#   20 <= T <= 65

x_rescaling = torch.tensor([120.0, 20.0]) # Must agree with first two constraints

def c_x(x, scale:float=1000, normalized_x:bool=normalize_x, x_rescaling:torch.Tensor=x_rescaling):
    x_copy = x.detach().clone()
    if normalized_x: x_copy = x_rescaling*x_copy
    A = torch.zeros(4,2)
    A[0,0] = -1.0
    A[1,0] =  1.0
    A[2,1] = -1.0
    A[3,1] =  1.0
    b = torch.tensor([0.0, -120.0, 0.0, -20.0])
    lin_ctrs = torch.matmul(A,x_copy)+b
    quad_ctrs = torch.prod(x_copy).unsqueeze(0)-500.0
    output = torch.cat((lin_ctrs, quad_ctrs), dim=0)
    return scale*output

def c_y(y, scale:float=1000):
    N = int(len(y)/6)
    output = torch.zeros(7*N)
    for i in range(N):
        output[7*i  ] = -y[6*i  ]
        output[7*i+1] = -y[6*i+1]
        output[7*i+2] = -y[6*i+2]
        output[7*i+3] = -y[6*i+3]
        output[7*i+4] = -y[6*i+4]
        output[7*i+5] = 20.0-y[6*i+5]
        output[7*i+6] = y[6*i+5]-65.0
    return scale*output


# y = Phi(x) and z = ReLU(c(x)), and yz = [y, z]
def  f_tilde(yz, m=m):
    y = yz[:m]
    z = yz[m:]
    return (f(y) - torch.linalg.norm(z)**2).item()

def df_tilde(yz, m=m):
    y = yz[:m]
    z = yz[m:]
    return torch.cat((df(y),-2*z), dim=0)



#%% Class generating the NN Phi

# Phi takes as input a two-dimensional vector x = (t,Q) (normalized, so x in [0,1]^2)
# and returns as output the transpose of the vector
#   [TG(t*i/N), DG(t*i/N), MG(t*i/N), G(t*i/N), ME(t*i/N), T(t*i/N), ... for all i in [[1,N]]]
class Phi_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = n
        self.N = N
        self.m = 6*N
        self.normalize_x = normalize_x
        self.inactive_subspace_f = inactive_subspace_f
        self.pinn = pinn
        self.rescale_x_layer = self.make_rescale_x_layer()
        self.map_to_pinn_input_vectorized = self.make_pinn_input_layer()

    def make_rescale_x_layer(self):
        layer = nn.Linear(self.n, self.n, bias=False)
        W = torch.eye(self.n)
        if self.normalize_x:
            W[0, 0] = x_rescaling[0]
            W[1, 1] = x_rescaling[1]
        layer.weight = nn.Parameter(W)
        return layer

    def make_pinn_input_layer(self):
        W = torch.zeros((2*self.N, 2))
        for i in range(self.N): W[2*i,0] = (i+1)/self.N; W[2*i+1,1] = 1
        layer = nn.Linear(self.n, 2*self.N, bias=False)
        layer.weight = nn.Parameter(W)
        return layer

    def forward(self, x):
        x_rescaled = self.rescale_x_layer(x)
        pinn_input_vectorized = self.map_to_pinn_input_vectorized(x_rescaled)
        pinn_input = torch.reshape(pinn_input_vectorized, (self.N, 2))
        pinn_output = self.pinn(pinn_input)
        pinn_output_vectorized = torch.reshape(pinn_output, (self.m, 1))
        if len(x.shape) == 1: pinn_output_vectorized = pinn_output_vectorized.squeeze(-1)
        return pinn_output_vectorized



#%% Class generating the NN Phi_tilde

class Phi_tilde_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.Phi = Phi
        self.n = self.Phi.n
        self.m = self.Phi.m + len(c_x(torch.zeros(self.Phi.n))) + len(c_y(Phi(torch.zeros(self.Phi.n))))
        self.inactive_subspace_f = self.Phi.inactive_subspace_f
        self.relu = nn.ReLU()

    def forward(self, x):
        xs = x.squeeze(0)
        y = self.Phi(xs)
        z_x = self.relu(c_x(xs))
        z_y = self.relu(c_y(y))
        out = torch.cat((y, z_x, z_y), dim=-1)
        if len(x.shape) > 1: out = out.unsqueeze(0)
        return out



#%% Generation of the NNs Phi and Phi_tilde

Phi = Phi_model().to(device).eval()
for param in Phi.parameters(): param.requires_grad = False
Phi_scripted = torch.jit.script(Phi)
path_save = "/".join([path_root, "Phi.pt"]); Phi_scripted.save(path_save)

Phi_tilde = Phi_tilde_model().to(device).eval()
for param in Phi_tilde.parameters(): param.requires_grad = False
Phi_tilde_scripted = torch.jit.script(Phi_tilde)
path_save = "/".join([path_root, "Phi_tilde.pt"]); Phi_tilde_scripted.save(path_save)



#%% Generation of the problem parameters

# Starting point and radius
x_0 = torch.tensor([40.0, 6.0])
if normalize_x: x_0 = x_0 * 1/x_rescaling
r_0 = 1E-3

# Minimal/maximal radius for the poll step and the attack step
r_dsm_min = 1E-6
r_atk_min = 1E-6
r_dsm_max = 1E-1
r_atk_max = 1E0

# Saving the parameters
parameters = [r_0, r_atk_min, r_atk_max, r_dsm_min, r_dsm_max]
path_save = "/".join([path_root, "x_0.pt"]);        torch.save(x_0, path_save)
path_save = "/".join([path_root, "parameters.pt"]); torch.save(parameters, path_save)

# List of points tested in preliminary study of attack algorithms
x_list_attack_analysis = [x_0]
for _ in range(99):
    x = torch.rand(n)
    x_list_attack_analysis.append(x)
path_save = "/".join([path_root, "attack_analysis_points.pt"]); torch.save(x_list_attack_analysis, path_save)
