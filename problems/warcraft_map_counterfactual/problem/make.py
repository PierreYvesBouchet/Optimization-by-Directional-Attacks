# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import numpy as np
import gurobipy as gp
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

# Problem-related packages
from build_data.dataloader        import get_dataloaders_and_dataset
from build_data.prediction_models import PartialResNet
from build_data.Vae               import Vae
from build_data.solve             import solve
from build_data.ShortestPathModel import ShortestPathModel



#%% Choice of the device to support the NN (hardcoded to CPU for portability)

if   torch.cuda.is_available():         device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else:                                   device = "cpu"
device = "cpu"



#%% Dimension of the VAE's latent space and of the CNN's output vector

# Cannot be changed, unless the VAE and the CNN are re-trained
n = 64
d = 144 # Number of pixels in CNN maps
m = n+d # Dimension of the output of Phi



#%% Import of the VAE and the CNN

# VAE import
vae = Vae(n)
path_vae_weights = "/".join([path_build_data, "VAE_weights.pt"])
vae.load_state_dict(torch.load(path_vae_weights, map_location=device, weights_only=True))
vae.eval()
for param in vae.parameters(): param.requires_grad = False
vae = vae.to(device)

# Maps --> latent space encoder (warcraft_map is a [0,1]^{3*96*96} tensor)
# Latent space --> maps decoder (z = vector in IR^n, the latent space of the VAE)
def encode(warcraft_map): return vae.encoder(warcraft_map)[0][0]
def decode(z)           : return vae.decoder(z)

# CNN import
k = int(np.sqrt(d))
cnn = PartialResNet(k=k)
path_cnn_weights = "/".join([path_build_data, "CNN_weights.pt"])
cnn.load_state_dict(torch.load(path_cnn_weights, map_location=device, weights_only=True))
cnn.eval()
for param in cnn.parameters(): param.requires_grad = False

# warcraft map --> cost to enter each block of 12 pixels (output is a vector in IR^m (m = 144, so = flattened 12*12 image))
def cost_map(warcraft_map): return cnn(warcraft_map).squeeze().detach()

# Function to alter the cost maps so that mountains and water lead to huge cost compared to land (c = (0,1,1) => no alteration of costs)
c0, c1, c2 = [0, 1, 1]
def cost_map_extended(cost_map): return c0+(cost_map/c1)**c2

# Compute the shortest path from top-left to bottom-right cost map (1st output is a {0,1}^144 vector (= flattened 12*12 bool array), 2nd output is the cost of the path)
def compute_shortest_path(cost_map):
    grid = (k,k)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with ShortestPathModel(grid, env=env, task="warcraft") as short_path_model: shortest_path = solve(cost_map, short_path_model, "warcraft")[0]
        return shortest_path

def compute_path_cost(cost_map, opt_path): return torch.sum(opt_path * cost_map.squeeze()).item()



#%% Import of the original image that is going to be altered and the target image for which the shortest path is the decision to explain
#       - initial_image, the starting image what will be altered by the explanation process
#       - target_image, the image for which we compute the shortest path that will be explained

# Load the dataset of images
(dataset_train_cnn, _, _, _, _, _) = get_dataloaders_and_dataset()

# Initial and target images indexes
# (index_initial, index_target) = (8743, 2677)
# (index_initial, index_target) = (5611,   12)
(index_initial, index_target) = (1839, 5892)

# Initial image
(map_initial, _, _, _) = dataset_train_cnn[index_initial]
z_initial        = encode(map_initial)
map_initial_encoded = decode(z_initial)
cost_map_initial = cost_map_extended(cost_map(map_initial_encoded))
opt_path_initial = compute_shortest_path(cost_map_initial)

# Target image
(map_target, _, _, _) = dataset_train_cnn[index_target]
z_target        = encode(map_target)
map_target_encoded = decode(z_target)
cost_map_target = cost_map_extended(cost_map(map_target_encoded))
opt_path_target = compute_shortest_path(cost_map_target)



#%% Goal and constraints functions

# Goal function to be maximized
def  f(y, x_0:torch.tensor=z_initial, y_0:torch.tensor=cost_map_initial, scale:float=100):
    yx = y[:64]
    yy = y[64:]
    output = 0
    output += -1*(torch.linalg.norm(yx-x_0).item()**2)
    output += -1*(( (torch.sum(yy)-torch.sum(y_0)) / scale ).item()**2)
    return output

# Components of the vector y that do not influence f
inactive_subspace_f = tuple([])

def df(y, x_0:torch.tensor=z_initial, y_0:torch.tensor=cost_map_initial, scale:float=100):
    yx = y[:64]
    yy = y[64:]
    output = torch.zeros_like(y)
    output[:64] = -2*(yx-x_0)
    output[64:] = -2/scale**2 * (1 + torch.sum(yy) - y_0)
    return output

# Constraint set F =
#   {x such that
#      ||x|| in [sqrt(n)+-1]
#      cost_path(path_target, Phi(x)) <= cost_path(path_target, Phi(x_target)) + epsilon*(cost_path(path_target, Phi(x_0))-cost_path(path_target, Phi(x_target)))
#   }

def c_x(x, dr:float=1.0, scale:float=10.0):
    n = torch.tensor(x.shape[-1])
    norm = torch.linalg.norm(x)
    delta = ( norm-(torch.sqrt(n)-dr) ) * ( norm-(torch.sqrt(n)+dr) )
    delta = delta.unsqueeze(0)/scale
    return delta

def c_y(y, p_ora, p_ref, eps:float=1.0, scale:float=10.0, m:int=64):
    c_ora = torch.sum(p_ora*y[m:])
    c_ref = torch.sum(p_ref*y[m:])
    value = c_ora*(1+eps)- c_ref
    value = value.unsqueeze(0)/scale
    return value



# y = Phi(x) and z = ReLU(c(x)), and yz = [y, z]
def f_tilde(yz, m=208):
    y = yz[:m]
    z = yz[m:]
    return (f(y) - torch.linalg.norm(z)**2).item()

def df_tilde(yz, m=208):
    y = yz[:m]
    z = yz[m:]
    return torch.cat((df(y),-2*z), dim=0)




#%% Class generating the NN Phi

class Phi_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.n = n
        self.m = m
        self.cnn = cnn
        self.vae = vae
        self.map_initial = map_initial
        self.map_target  = map_target
        self.z_initial = encode(self.map_initial).clone().detach()
        self.z_target  = encode(self.map_target ).clone().detach()
        self.cost_expansion_coeffs = [c0,c1,c2] # the CNN map_cost becomes coeff0 + (map_cost/coeff1)**coeff2
        self.cost_initial = self.cost_expansion_coeffs[0] + (self.cnn(self.map_initial.unsqueeze(0)).squeeze(0)/self.cost_expansion_coeffs[1])**self.cost_expansion_coeffs[2]
        self.cost_target  = self.cost_expansion_coeffs[0] + (self.cnn(self.map_target.unsqueeze( 0)).squeeze(0)/self.cost_expansion_coeffs[1])**self.cost_expansion_coeffs[2]
        self.opt_path_initial = compute_shortest_path(self.cost_initial).clone().detach()
        self.opt_path_target  = compute_shortest_path(self.cost_target ).clone().detach()
        self.inactive_subspace_f = inactive_subspace_f

    def forward(self, x):
        xs = x.squeeze(0)
        warcraft_map = self.vae.decoder(xs)
        cost_map = self.cnn(warcraft_map)
        cost_map_extended = self.cost_expansion_coeffs[0] + (cost_map/self.cost_expansion_coeffs[1])**self.cost_expansion_coeffs[2]
        if len(x.shape) == 1: cost_map_extended = cost_map_extended.squeeze(0)
        out = torch.cat((x, cost_map_extended), dim=-1)
        if len(x.shape) > 1: out = out.unsqueeze(0)
        return out



#%% Class generating the NN Phi_tilde

class Phi_tilde_model(nn.Module):

    def __init__(self, Phi):
        super().__init__()
        self.Phi = Phi
        self.n = self.Phi.n
        self.m = self.Phi.m + len(c_x(torch.zeros(self.Phi.n))) + len(c_y(torch.zeros(self.Phi.m), self.Phi.opt_path_target, self.Phi.opt_path_initial))
        self.inactive_subspace_f = self.Phi.inactive_subspace_f
        self.relu = nn.ReLU()

    def forward(self, x):
        xs = x.squeeze(0)
        y = self.Phi(xs)
        z_x = self.relu(c_x(xs))
        z_y = self.relu(c_y(y, self.Phi.opt_path_target, self.Phi.opt_path_initial))
        out = torch.cat((y, z_x, z_y), dim=-1)
        if len(x.shape) > 1: out = out.unsqueeze(0)
        return out



#%% Generation of the NNs Phi and Phi_tilde

Phi = Phi_model().to(device).eval()
for param in Phi.parameters(): param.requires_grad = False
Phi_scripted = torch.jit.script(Phi)
path_save = "/".join([path_root, "Phi.pt"]); Phi_scripted.save(path_save)

Phi_tilde = Phi_tilde_model(Phi).to(device).eval()
for param in Phi_tilde.parameters(): param.requires_grad = False
Phi_tilde_scripted = torch.jit.script(Phi_tilde)
path_save = "/".join([path_root, "Phi_tilde.pt"]); Phi_tilde_scripted.save(path_save)



#%% Generation of the problem parameters

# Saves the CNN+cost_extension output of the reference image (since it is the target in f)
path_save = "/".join([path_root, "cost_map_initial.pt"]); torch.save(cost_map_initial, path_save)

# Starting point and radius
x_0 = z_initial
r_0 = 1E0

# Minimal/maximal radius for the poll step and the attack step
r_dsm_min = 1E-4
r_atk_min = 1E-5
r_dsm_max = 2E2
r_atk_max = 5E0

# Saving the parameters
parameters = [r_0, r_atk_min, r_atk_max, r_dsm_min, r_dsm_max]
path_save = "/".join([path_root, "x_0.pt"]);        torch.save(x_0, path_save)
path_save = "/".join([path_root, "parameters.pt"]); torch.save(parameters, path_save)

# List of points tested in preliminary study of attack algorithms
x_list_attack_analysis = [x_0, z_target]
for _ in range(98):
    i = np.random.randint(0, len(dataset_train_cnn))
    x = encode(dataset_train_cnn[i][0])
    x_list_attack_analysis.append(x)
path_save = "/".join([path_root, "attack_analysis_points.pt"]); torch.save(x_list_attack_analysis, path_save)



#%% Plot of the shortest path in both images

do_plots = False

# Functions to ease the plots
def convert_map_to_print(warcract_map): return warcract_map.numpy().transpose((1, 2, 0))
def convert_cost_to_print(cost_map)   : return nn.Unflatten(0, (k,k))(cost_map)
def convert_map_plus_path_to_print(warcract_map, opt_path, alpha=0.25):
    map_plus_path_array = convert_map_to_print(warcract_map).copy()
    shortest_path_array = nn.Unflatten(0, (k,k))(opt_path)
    for x in range(96):
        for y in range(96):
            cell = map_plus_path_array[x,y,:]
            if shortest_path_array[int(x/8), int(y/8)] == 0.0: cell = (1-alpha)*cell
            else:                                              cell = cell + (1-cell)*alpha
            map_plus_path_array[x,y,:] = cell
    return map_plus_path_array

# Plot both images and their associated shortest path
if do_plots:
    fig, ax = plt.subplots(ncols=7)
    for i in range(7): ax[i].axis("off")

    ax[0].imshow(convert_map_to_print(map_initial_encoded.squeeze(0)))
    ax[0].set_title(r"$\mathcal{W}_\mathrm{ini} \triangleq \mathrm{warcraft}(x_\mathrm{ini})$")
    ax[1].imshow(convert_cost_to_print(cost_map_initial))
    ax[1].set_title(r"$\mathrm{costmap}(\mathcal{W}_\mathrm{ini})$")
    ax[2].imshow(convert_map_plus_path_to_print(map_initial_encoded.squeeze(0), opt_path_initial))
    ax[2].set_title(r"optimal path $\mathrm{p}^*_\mathrm{ini}$")
    ax[3].imshow(convert_map_plus_path_to_print(map_initial_encoded.squeeze(0), opt_path_target))
    ax[3].set_title(r"alternative path $\mathrm{p}^\sharp$")
    ax[4].imshow(convert_map_to_print(map_target_encoded.squeeze(0)))
    ax[4].set_title(r"$\mathcal{W}_\mathrm{cfa} \triangleq \mathrm{warcraft}(x_\mathrm{cfa})$")
    ax[5].imshow(convert_cost_to_print(cost_map_target))
    ax[5].set_title(r"$\mathrm{costmap}(\mathcal{W}_\mathrm{cfa}))$")
    ax[6].imshow(convert_map_plus_path_to_print(map_target_encoded.squeeze(0), opt_path_target))
    ax[6].set_title(r"alternative path $\mathrm{p}^\sharp$")

    # ax[0].imshow(convert_map_to_print(map_initial))
    # ax[0].set_title("Real Warcraft map")
    # ax[1].imshow(convert_cost_to_print(cost_map_initial))
    # ax[1].set_title("Travel cost map")
    # ax[2].imshow(convert_map_plus_path_to_print(map_initial, opt_path_initial))
    # ax[2].set_title(r"Shortest path")
    # ax[3].imshow(convert_map_to_print(map_target_encoded.squeeze(0)))
    # ax[3].set_title("Generated map")
    # ax[4].imshow(convert_cost_to_print(cost_map_target))
    # ax[4].set_title("Travel cost map")
    # ax[5].imshow(convert_map_plus_path_to_print(map_target_encoded.squeeze(0), opt_path_target))
    # ax[5].set_title(r"Shortest path")
    fig.set_size_inches(12,2)
    fig.tight_layout()
    fig.savefig("/".join([path_root, "example_maps.pdf"]))
