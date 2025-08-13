# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import gurobipy as gp
import os
import sys
import matplotlib.pyplot as plt
plt.close("all")

# Torch-related packages
import torch
from torch import nn

path_root = os.path.dirname(os.path.abspath(__file__))
path_problem = "/".join([path_root, "..", "problem"])
sys.path.append(path_problem)

# Problem-related packages
from build_data.solve import solve
from build_data.ShortestPathModel import ShortestPathModel



#%%

Phi_tilde = torch.jit.load("/".join([path_problem, "Phi_tilde.pt"]))
Phi = Phi_tilde.Phi

result_file_atk = torch.load("/".join([path_root, "optim_attacks.pt"]),       weights_only=True)
result_file_rls = torch.load("/".join([path_root, "optim_line_searches.pt"]), weights_only=True)
result_file_hyb = torch.load("/".join([path_root, "optim_hybrid.pt"]),        weights_only=True)
result_file_dsm = torch.load("/".join([path_root, "optim_dsm.pt"]),           weights_only=True)

vae = Phi.vae
cnn = Phi.cnn
warcraft_to_cost_map_extended = lambda warcraft_map: Phi.cost_expansion_coeffs[0] + (Phi.cnn(warcraft_map).squeeze(0)/Phi.cost_expansion_coeffs[1])**Phi.cost_expansion_coeffs[2]

z_ini = Phi.z_initial
z_tar = Phi.z_target
z_atk = result_file_atk[-1][0]
z_rls = result_file_rls[-1][0]
z_hyb = result_file_hyb[-1][0]
z_dsm = result_file_dsm[-1][0]

map_ini = Phi.vae.decoder(z_ini).squeeze(0) # Phi.map_initial
map_tar = Phi.vae.decoder(z_tar).squeeze(0) # Phi.map_target
map_atk = Phi.vae.decoder(z_atk).squeeze(0)
map_rls = Phi.vae.decoder(z_rls).squeeze(0)
map_hyb = Phi.vae.decoder(z_hyb).squeeze(0)
map_dsm = Phi.vae.decoder(z_dsm).squeeze(0)

cost_map_ini = warcraft_to_cost_map_extended(map_ini.unsqueeze(0)).squeeze().detach()
cost_map_tar = warcraft_to_cost_map_extended(map_tar.unsqueeze(0)).squeeze().detach()
cost_map_atk = warcraft_to_cost_map_extended(map_atk.unsqueeze(0)).squeeze().detach()
cost_map_rls = warcraft_to_cost_map_extended(map_rls.unsqueeze(0)).squeeze().detach()
cost_map_hyb = warcraft_to_cost_map_extended(map_hyb.unsqueeze(0)).squeeze().detach()
cost_map_dsm = warcraft_to_cost_map_extended(map_dsm.unsqueeze(0)).squeeze().detach()

cost_target_path_on_map_atk = result_file_atk[-1][1]
cost_target_path_on_map_rls = result_file_rls[-1][1]
cost_target_path_on_map_hyb = result_file_hyb[-1][1]
cost_target_path_on_map_dsm = result_file_dsm[-1][1]



#%%

def compute_shortest_path(cost_map):
    grid = (12, 12)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with ShortestPathModel(grid, env=env, task="warcraft") as short_path_model: shortest_path, cost_path = solve(cost_map, short_path_model, "warcraft")
        return shortest_path, cost_path

opt_path_ini, cost_opt_path_ini = compute_shortest_path(cost_map_ini)
opt_path_tar, cost_opt_path_tar = compute_shortest_path(cost_map_tar)
opt_path_atk, cost_opt_path_atk = compute_shortest_path(cost_map_atk)
opt_path_rls, cost_opt_path_rls = compute_shortest_path(cost_map_rls)
opt_path_hyb, cost_opt_path_hyb = compute_shortest_path(cost_map_hyb)
opt_path_dsm, cost_opt_path_dsm = compute_shortest_path(cost_map_dsm)

def compute_path_cost(cost_map, opt_path): return torch.sum(opt_path * cost_map.squeeze()).item()



#%%

def print_map(warcract_map): return warcract_map.numpy().transpose((1, 2, 0))

def print_cost_map(costs):   return nn.Unflatten(0, (12, 12))(costs)

def print_map_plus_path(warcract_map, shortest_path, alpha=0.25):
    map_plus_path_array = print_map(warcract_map).copy()
    shortest_path_array = nn.Unflatten(0, (12, 12))(shortest_path)
    for x in range(96):
        for y in range(96):
            cell = map_plus_path_array[x,y,:]
            if shortest_path_array[int(x/8), int(y/8)] == 0.0: cell = (1-alpha)*cell
            else:                                              cell = cell + (1-cell)*alpha
            map_plus_path_array[x,y,:] = cell
    return map_plus_path_array

def print_cost_map_plus_path(cost_map, shortest_path, alpha=0.25):
    max_cell_value = torch.max(cost_map)
    map_plus_path_array = print_cost_map(cost_map).clone()
    shortest_path_array = print_cost_map(shortest_path)
    for x in range(12):
        for y in range(12):
            cell = map_plus_path_array[x,y]/max_cell_value
            if shortest_path_array[x,y] == 0.0: cell = cell#(1-alpha)*cell
            else:                               cell = -float("inf")
            map_plus_path_array[x,y] = cell
    return map_plus_path_array

def format_cost_for_title(cost_map, path, x_name, p_name):
    cost = compute_path_cost(cost_map, path)
    s = r"$\mathrm{cost} = " + "{:>6.3f}".format(cost) + "$"
    # s = r"$\mathrm{costmap}(" + x_name + ", " + p_name + ") = " + "{:>6.3f}".format(cost) + "$"
    # s = "".join(["$", "\mathrm{costmap}", "(", x_name, ",", p_name, ") = ", "{:>6.3f}".format(cost), "$"])
    return s

fig, axes = plt.subplots(3, 10, figsize=(18, 5))
# Plots of the raw images
axes[0,0].imshow(print_map(map_ini));           axes[0,0].set_title(r"$\mathrm{warcraft}(x_{ini})$")
axes[0,2].imshow(print_map(map_rls));           axes[0,2].set_title(r"$\mathrm{warcraft}(x^*_{rls})$")
axes[0,4].imshow(print_map(map_dsm));           axes[0,4].set_title(r"$\mathrm{warcraft}(x^*_{dsm})$")
axes[0,6].imshow(print_map(map_atk));           axes[0,6].set_title(r"$\mathrm{warcraft}(x^*_{atk})$")
axes[0,8].imshow(print_map(map_hyb));           axes[0,8].set_title(r"$\mathrm{warcraft}(x^*_{hyb})$")
axes[0,1].imshow(print_cost_map(cost_map_ini)); axes[0,1].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x_{ini}))$")
axes[0,3].imshow(print_cost_map(cost_map_rls)); axes[0,3].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{rls}))$")
axes[0,5].imshow(print_cost_map(cost_map_dsm)); axes[0,5].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{dsm}))$")
axes[0,7].imshow(print_cost_map(cost_map_atk)); axes[0,7].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{atk}))$")
axes[0,9].imshow(print_cost_map(cost_map_hyb)); axes[0,9].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{hyb}))$")
# Plots of the initial path for each image
axes[1,0].imshow(print_map_plus_path(map_ini, opt_path_ini));           axes[1,0].set_title(r"$+ \mathrm{p}^*_{ini}$")
axes[1,2].imshow(print_map_plus_path(map_rls, opt_path_ini));           axes[1,2].set_title(r"$+ \mathrm{p}^*_{ini}$")
axes[1,4].imshow(print_map_plus_path(map_dsm, opt_path_ini));           axes[1,4].set_title(r"$+ \mathrm{p}^*_{ini}$")
axes[1,6].imshow(print_map_plus_path(map_atk, opt_path_ini));           axes[1,6].set_title(r"$+ \mathrm{p}^*_{ini}$")
axes[1,8].imshow(print_map_plus_path(map_hyb, opt_path_ini));           axes[1,8].set_title(r"$+ \mathrm{p}^*_{ini}$")
axes[1,1].imshow(print_cost_map_plus_path(cost_map_ini, opt_path_ini)); axes[1,1].set_title(format_cost_for_title(cost_map_ini, opt_path_ini, "x_{ini}",   "\mathrm{p}^*_{ini}"))
axes[1,3].imshow(print_cost_map_plus_path(cost_map_rls, opt_path_ini)); axes[1,3].set_title(format_cost_for_title(cost_map_rls, opt_path_ini, "x^*_{rls}", "\mathrm{p}^*_{ini}"))
axes[1,5].imshow(print_cost_map_plus_path(cost_map_dsm, opt_path_ini)); axes[1,5].set_title(format_cost_for_title(cost_map_dsm, opt_path_ini, "x^*_{dsm}", "\mathrm{p}^*_{ini}"))
axes[1,7].imshow(print_cost_map_plus_path(cost_map_atk, opt_path_ini)); axes[1,7].set_title(format_cost_for_title(cost_map_atk, opt_path_ini, "x^*_{atk}", "\mathrm{p}^*_{ini}"))
axes[1,9].imshow(print_cost_map_plus_path(cost_map_hyb, opt_path_ini)); axes[1,9].set_title(format_cost_for_title(cost_map_hyb, opt_path_ini, "x^*_{hyb}", "\mathrm{p}^*_{ini}"))
# Plots of the target path on each image
axes[2,0].imshow(print_map_plus_path(map_ini, opt_path_tar)); axes[2,0].set_title(r"$+ \mathrm{p}^\sharp$")
axes[2,2].imshow(print_map_plus_path(map_rls, opt_path_tar)); axes[2,2].set_title(r"$+ \mathrm{p}^\sharp$")
axes[2,4].imshow(print_map_plus_path(map_dsm, opt_path_tar)); axes[2,4].set_title(r"$+ \mathrm{p}^\sharp$")
axes[2,6].imshow(print_map_plus_path(map_atk, opt_path_tar)); axes[2,6].set_title(r"$+ \mathrm{p}^\sharp$")
axes[2,8].imshow(print_map_plus_path(map_hyb, opt_path_tar)); axes[2,8].set_title(r"$+ \mathrm{p}^\sharp$")
axes[2,1].imshow(print_cost_map_plus_path(cost_map_ini, opt_path_tar)); axes[2,1].set_title(format_cost_for_title(cost_map_ini, opt_path_tar, "x_{ini}",   "\mathrm{p}^\sharp"))
axes[2,3].imshow(print_cost_map_plus_path(cost_map_rls, opt_path_tar)); axes[2,3].set_title(format_cost_for_title(cost_map_rls, opt_path_tar, "x^*_{rls}", "\mathrm{p}^\sharp"))
axes[2,5].imshow(print_cost_map_plus_path(cost_map_dsm, opt_path_tar)); axes[2,5].set_title(format_cost_for_title(cost_map_dsm, opt_path_tar, "x^*_{dsm}", "\mathrm{p}^\sharp"))
axes[2,7].imshow(print_cost_map_plus_path(cost_map_atk, opt_path_tar)); axes[2,7].set_title(format_cost_for_title(cost_map_atk, opt_path_tar, "x^*_{atk}", "\mathrm{p}^\sharp"))
axes[2,9].imshow(print_cost_map_plus_path(cost_map_hyb, opt_path_tar)); axes[2,9].set_title(format_cost_for_title(cost_map_hyb, opt_path_tar, "x^*_{hyb}", "\mathrm{p}^\sharp"))

for i in range(3):
    for j in range(10):
        # axes[i,j].axis("off")
        # axes[i,j].set_frame_on(True)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])

fig.tight_layout()
fig.subplots_adjust(wspace=0.20, hspace=0.20, top=0.95, bottom=0.01, left=0.01, right=0.99)

fig.savefig("/".join([path_root, "results_maps.pdf"]))
