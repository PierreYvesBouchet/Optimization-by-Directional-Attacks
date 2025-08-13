# -*- coding: utf-8 -*-



#%% Libraries import

# Torch-related packages
import torch
from torch import nn

# Project-dependent files
from build_data.dataloader import get_dataloaders_and_dataset
from build_data.prediction_models import PartialResNet
from build_data.Vae import Vae
from build_data.solve import solve
from build_data.ShortestPathModel import ShortestPathModel

# Generic Python packages
import gurobipy as gp
import numpy as np
import matplotlib.pyplot as plt
import random
plt.close("all")



#%% Choice of the dimension of the VAE's latent space and of the CNN's output vector

# If n =/= 64, then the weights of the VAE cannot be loaded from current file
n = 64

# if m =/= 144, then the weights of the CNN cannot be loaded from current file
m = 144

# Fixing the seed for reproducibility
torch.manual_seed(0)



#%% Choice of the device to support the NN (hardcoded to CPU for portability)

if   torch.cuda.is_available():         device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else:                                   device = "cpu"
device = "cpu" # Delete this line for optimized hardware selection



#%% Import of the VAE and the CNN

vae = Vae(n)
vae.load_state_dict(torch.load("build_data/VAE_weights.pt", map_location=device, weights_only=True))
vae.eval()
for param in vae.parameters(): param.requires_grad = False
vae = vae.to("cpu")

k = int(np.sqrt(m))
cnn = PartialResNet(k=k)
cnn.load_state_dict(torch.load("build_data/CNN_weights.pt", map_location=device, weights_only=True))
cnn.eval()
for param in cnn.parameters(): param.requires_grad = False

# Function to modify costs computed by the CNN, if desired; (c0,c1,c2) = (0,1,1) => no change
def expand_cost(map_cost, c0=0, c1=1, c2=1): return c0+(map_cost/c1)**c2



#%%

(dataset_train_cnn, _, _, _, _, _) = get_dataloaders_and_dataset()

def compute_shortest_path(map_costs):
    grid = (12, 12)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with ShortestPathModel(grid, env=env, task="warcraft") as short_path_model: shortest_path, cost_path = solve(map_costs, short_path_model, "warcraft")
        return shortest_path, cost_path

def convert_map_to_print(warcract_map): return warcract_map.numpy().transpose((1, 2, 0))

def convert_cost_to_print(costs)      : return nn.Unflatten(0, (12, 12))(costs)

def convert_map_plus_path_to_print(warcract_map, shortest_path, alpha=0.25):
    map_plus_path_array = convert_map_to_print(warcract_map).copy()
    shortest_path_array = nn.Unflatten(0, (12, 12))(shortest_path)
    for x in range(96):
        for y in range(96):
            cell = map_plus_path_array[x,y,:]
            if shortest_path_array[int(x/8), int(y/8)] == 0.0: cell = (1-alpha)*cell
            else:                                              cell = cell + (1-cell)*alpha
            map_plus_path_array[x,y,:] = cell
    return map_plus_path_array

def convert_cost_map_plus_path_to_print(cost_map, shortest_path, alpha=0.25):
    max_cell_value = torch.max(cost_map)
    map_plus_path_array = convert_cost_to_print(cost_map).clone()
    shortest_path_array = nn.Unflatten(0, (12, 12))(shortest_path)
    for x in range(12):
        for y in range(12):
            cell = map_plus_path_array[x,y]/max_cell_value
            if shortest_path_array[x,y] == 0.0: cell = (1-alpha)*cell
            else:                               cell = 1#cell + (1-cell)*alpha
            map_plus_path_array[x,y] = cell#
    return map_plus_path_array



#%%

N = 5
fig, axes = plt.subplots(2, N, figsize=(12, 5))
for i in range(N):

    # Initial and target images indexes
    indice = random.sample(range(0, 10000), 1)[0]

    (warcraft_map, _, _, _) = dataset_train_cnn[indice]
    warcraft_map_encoded = vae.encoder(warcraft_map)[0][0]
    warcraft_map_costs   = cnn(warcraft_map.unsqueeze(0)).squeeze().detach()
    warcraft_map_costs   = expand_cost(warcraft_map_costs)
    warcraft_map_shortest_path, _ = compute_shortest_path(warcraft_map_costs)

    axes[0,i].imshow(convert_map_to_print(warcraft_map))
    axes[1,i].imshow(convert_cost_map_plus_path_to_print(warcraft_map_costs, warcraft_map_shortest_path))
    axes[0,i].set_title(indice)
    axes[1,i].set_title(indice)
    axes[0,i].axis('off')
    axes[1,i].axis('off')

plt.tight_layout()
plt.show()



#%% Plots some nice maps we've found in the database

nice_maps_indices = [   2,   12,   64,  116,  418, 1200, 1839, 2188, 2666, 2677, 2524, 3091, 3181, 3292, 3716,  588,
                     3975, 4437, 4631, 5892, 5904, 6476, 6705, 7297, 7536, 8743, 8819, 8822, 9316, 9359, 9641, 5611]

nice_maps_indices.sort()

fig, axes = plt.subplots(4, len(nice_maps_indices)//2, figsize=(len(nice_maps_indices)//2, 6))
for i, indice in enumerate(nice_maps_indices):

    (warcraft_map, _, _, _) = dataset_train_cnn[indice]
    warcraft_map_encoded = vae.encoder(warcraft_map)[0][0]
    warcraft_map_costs   = cnn(warcraft_map.unsqueeze(0)).squeeze().detach()
    warcraft_map_costs   = expand_cost(warcraft_map_costs)
    warcraft_map_shortest_path, _ = compute_shortest_path(warcraft_map_costs)

    lin = 2*(i %  2)
    col = i // 2
    axes[lin  , col].imshow(convert_map_to_print(warcraft_map))
    axes[lin+1, col].imshow(convert_cost_map_plus_path_to_print(warcraft_map_costs, warcraft_map_shortest_path))
    axes[lin  , col].set_title(indice)
    axes[lin  , col].axis('off')
    axes[lin+1, col].axis('off')

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
plt.tight_layout()
plt.show()



#%%

def distance_between_maps_from_indices(i1, i2):
    m1 = dataset_train_cnn[i1][0]
    m2 = dataset_train_cnn[i2][0]
    z1 = vae.encoder(m1)[0][0]
    z2 = vae.encoder(m2)[0][0]
    print(torch.norm(z2-z1))



#%% Temp section for JOpts

def compute_path_cost(cost_map, opt_path): return torch.sum(opt_path * cost_map.squeeze()).item()

indice_1 = 8743
indice_2 = 2677
indices = [indice_1, indice_2]

fig, axes = plt.subplots(2, 3, figsize=(8,6))
for i, indice in enumerate(indices):

    (warcraft_map, _, _, _) = dataset_train_cnn[indice]
    warcraft_map_encoded = vae.encoder(warcraft_map)[0][0]
    warcraft_map_costs   = cnn(warcraft_map.unsqueeze(0)).squeeze().detach()
    warcraft_map_costs   = expand_cost(warcraft_map_costs)
    warcraft_map_shortest_path, cost = compute_shortest_path(warcraft_map_costs)

    axes[i,0].imshow(convert_map_to_print(warcraft_map))
    axes[i,1].imshow(convert_cost_to_print(warcraft_map_costs))
    axes[i,2].imshow(convert_cost_map_plus_path_to_print(warcraft_map_costs, warcraft_map_shortest_path))
    axes[i,0].set_title("map {:d}".format(indice))
    axes[i,1].set_title("CNN-predicted travel cost")
    axes[i,2].set_title("Shortest path, cost = {:.3f}".format(cost))
    for j in range(3): axes[i,j].axis("off")

fig.subplots_adjust(hspace=0.0)
fig.tight_layout()
fig.savefig("JOpts_illustration_1.pdf")



fig, axes = plt.subplots(1, 4, figsize=(12,3))

map_1 = dataset_train_cnn[indice_1][0]; map_costs_1 = expand_cost(cnn(map_1.unsqueeze(0)).squeeze().detach()); path_1 = compute_shortest_path(map_costs_1)[0]; cost_1 = compute_path_cost(map_costs_1, path_1)
map_2 = dataset_train_cnn[indice_2][0]; map_costs_2 = expand_cost(cnn(map_2.unsqueeze(0)).squeeze().detach()); path_2 = compute_shortest_path(map_costs_2)[0]; cost_2 = compute_path_cost(map_costs_1, path_2)

axes[0].imshow(convert_map_to_print(map_1));                              axes[0].axis("off"); axes[0].set_title("map {:d}".format(indice_1))
axes[1].imshow(convert_cost_to_print(map_costs_1));                       axes[1].axis("off"); axes[1].set_title("CNN-predicted travel cost")
axes[2].imshow(convert_cost_map_plus_path_to_print(map_costs_1, path_1)); axes[2].axis("off"); axes[2].set_title("Shortest path, cost = {:.3f}".format(cost_1))
axes[3].imshow(convert_cost_map_plus_path_to_print(map_costs_1, path_2)); axes[3].axis("off"); axes[3].set_title("Suggested path, cost = {:.3f}".format(cost_2))

fig.tight_layout()
fig.savefig("JOpts_illustration_2.pdf")
