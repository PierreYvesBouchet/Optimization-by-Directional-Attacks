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

i_best_hybr = 0
obj_best_hybr = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_hybrid(FFGSM)_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_hybr: i_best_hybr = i; obj_best_hybr = obj
result_file_hybr = torch.load("/".join([path_root, "optim_hybrid(FFGSM)_run"+str(i_best_hybr)+".pt"]), weights_only=True)

i_best_atck = 0
obj_best_atck = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_attacks(RFGSM)_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_hybr: i_best_atck = i; obj_best_atck = obj
result_file_atck = torch.load("/".join([path_root, "optim_attacks(RFGSM)_run"+str(i_best_atck)+".pt"]), weights_only=True)

i_best_back = 0
obj_best_back = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_backprop_ascent_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_back: i_best_atck = i; obj_best_back = obj
result_file_back = torch.load("/".join([path_root, "optim_backprop_ascent_run"+str(i_best_back)+".pt"]), weights_only=True)

i_best_cdsm = 0
obj_best_cdsm = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_dsm_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_cdsm: i_best_cdsm = i; obj_best_cdsm = obj
result_file_cdsm = torch.load("/".join([path_root, "optim_dsm_run"+str(i_best_cdsm)+".pt"]), weights_only=True)

i_best_brls = 0
obj_best_brls = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_line_searches_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_brls: i_best_brls = i; obj_best_brls = obj
result_file_brls = torch.load("/".join([path_root, "optim_line_searches_run"+str(i_best_brls)+".pt"]), weights_only=True)

i_best_bfgs = 0
obj_best_bfgs = -float("inf")
for i in range(5):
    results = torch.load("/".join([path_root, "optim_bfgs_run"+str(i)+".pt"]), weights_only=True)
    obj = results[-1][1]
    if obj > obj_best_bfgs: i_best_bfgs = i; obj_best_bfgs = obj
result_file_bfgs = torch.load("/".join([path_root, "optim_bfgs_run"+str(i_best_bfgs)+".pt"]), weights_only=True)

# result_file_hybr = torch.load("/".join([path_root, "optim_hybrid_run2.pt"]),         weights_only=True)
# result_file_atck = torch.load("/".join([path_root, "optim_attacks_run2.pt"]),        weights_only=True)
# result_file_back = torch.load("/".join([path_root, "optim_backprop_ascent_run2.pt"]),weights_only=True)
# result_file_cdsm = torch.load("/".join([path_root, "optim_dsm_run2.pt"]),            weights_only=True)
# result_file_brls = torch.load("/".join([path_root, "optim_line_searches_run2.pt"]),  weights_only=True)
# result_file_bfgs = torch.load("/".join([path_root, "optim_bfgs_run2.pt"]),           weights_only=True)



#%%

Phi_tilde = torch.jit.load("/".join([path_problem, "Phi_tilde.pt"]))
Phi = Phi_tilde.Phi

vae = Phi.vae
cnn = Phi.cnn
warcraft_to_cost_map_extended = lambda warcraft_map: Phi.cost_expansion_coeffs[0] + (Phi.cnn(warcraft_map).squeeze(0)/Phi.cost_expansion_coeffs[1])**Phi.cost_expansion_coeffs[2]

z_init = Phi.z_initial
z_targ = Phi.z_target
z_hybr = result_file_hybr[-1][0]
z_atck = result_file_atck[-1][0]
z_back = result_file_back[-1][0]
z_brls = result_file_brls[-1][0]
z_cdsm = result_file_cdsm[-1][0]
z_bfgs = result_file_bfgs[-1][0]

map_init = Phi.vae.decoder(z_init).squeeze(0) # Phi.map_initial
map_targ = Phi.vae.decoder(z_targ).squeeze(0) # Phi.map_target
map_hybr = Phi.vae.decoder(z_hybr).squeeze(0)
map_atck = Phi.vae.decoder(z_atck).squeeze(0)
map_back = Phi.vae.decoder(z_back).squeeze(0)
map_brls = Phi.vae.decoder(z_brls).squeeze(0)
map_cdsm = Phi.vae.decoder(z_cdsm).squeeze(0)
map_bfgs = Phi.vae.decoder(z_bfgs).squeeze(0)

cost_map_init = warcraft_to_cost_map_extended(map_init.unsqueeze(0)).squeeze().detach()
cost_map_targ = warcraft_to_cost_map_extended(map_targ.unsqueeze(0)).squeeze().detach()
cost_map_hybr = warcraft_to_cost_map_extended(map_hybr.unsqueeze(0)).squeeze().detach()
cost_map_atck = warcraft_to_cost_map_extended(map_atck.unsqueeze(0)).squeeze().detach()
cost_map_back = warcraft_to_cost_map_extended(map_back.unsqueeze(0)).squeeze().detach()
cost_map_brls = warcraft_to_cost_map_extended(map_brls.unsqueeze(0)).squeeze().detach()
cost_map_cdsm = warcraft_to_cost_map_extended(map_cdsm.unsqueeze(0)).squeeze().detach()
cost_map_bfgs = warcraft_to_cost_map_extended(map_bfgs.unsqueeze(0)).squeeze().detach()

cost_target_path_on_map_hybr = result_file_hybr[-1][1]
cost_target_path_on_map_atck = result_file_atck[-1][1]
cost_target_path_on_map_back = result_file_back[-1][1]
cost_target_path_on_map_brls = result_file_brls[-1][1]
cost_target_path_on_map_cdsm = result_file_cdsm[-1][1]
cost_target_path_on_map_bfgs = result_file_bfgs[-1][1]



#%%

def compute_shortest_path(cost_map):
    grid = (12, 12)
    with gp.Env(empty=True) as env:
        env.setParam('OutputFlag', 0)
        env.start()
        with ShortestPathModel(grid, env=env, task="warcraft") as short_path_model: shortest_path, cost_path = solve(cost_map, short_path_model, "warcraft")
        return shortest_path, cost_path

opt_path_init, cost_opt_path_init = compute_shortest_path(cost_map_init)
opt_path_targ, cost_opt_path_targ = compute_shortest_path(cost_map_targ)
opt_path_hybr, cost_opt_path_hybr = compute_shortest_path(cost_map_hybr)
opt_path_atck, cost_opt_path_atck = compute_shortest_path(cost_map_atck)
opt_path_back, cost_opt_path_back = compute_shortest_path(cost_map_back)
opt_path_brls, cost_opt_path_brls = compute_shortest_path(cost_map_brls)
opt_path_cdsm, cost_opt_path_cdsm = compute_shortest_path(cost_map_cdsm)
opt_path_bfgs, cost_opt_path_bfgs = compute_shortest_path(cost_map_bfgs)

def compute_path_cost(cost_map, opt_path): return torch.sum(opt_path * cost_map.squeeze()).item()



#%%

# def print_map(warcract_map): return warcract_map.numpy().transpose((1, 2, 0))

# def print_cost_map(costs):   return nn.Unflatten(0, (12, 12))(costs)

# def print_map_plus_path(warcract_map, shortest_path, alpha=0.25):
#     map_plus_path_array = print_map(warcract_map).copy()
#     shortest_path_array = nn.Unflatten(0, (12, 12))(shortest_path)
#     for x in range(96):
#         for y in range(96):
#             cell = map_plus_path_array[x,y,:]
#             if shortest_path_array[int(x/8), int(y/8)] == 0.0: cell = (1-alpha)*cell
#             else:                                              cell = cell + (1-cell)*alpha
#             map_plus_path_array[x,y,:] = cell
#     return map_plus_path_array

# def print_cost_map_plus_path(cost_map, shortest_path, alpha=0.25):
#     max_cell_value = torch.max(cost_map)
#     map_plus_path_array = print_cost_map(cost_map).clone()
#     shortest_path_array = print_cost_map(shortest_path)
#     for x in range(12):
#         for y in range(12):
#             cell = map_plus_path_array[x,y]/max_cell_value
#             if shortest_path_array[x,y] == 0.0: cell = cell#(1-alpha)*cell
#             else:                               cell = -float("inf")
#             map_plus_path_array[x,y] = cell
#     return map_plus_path_array

# def format_cost_for_title(cost_map, path, x_name, p_name):
#     cost = compute_path_cost(cost_map, path)
#     s = r"$\mathrm{cost} = " + "{:>6.3f}".format(cost) + "$"
#     # s = r"$\mathrm{costmap}(" + x_name + ", " + p_name + ") = " + "{:>6.3f}".format(cost) + "$"
#     # s = "".join(["$", "\mathrm{costmap}", "(", x_name, ",", p_name, ") = ", "{:>6.3f}".format(cost), "$"])
#     return s

# fig, axes = plt.subplots(3, 10, figsize=(18, 5))
# # Plots of the raw images
# axes[0,0].imshow(print_map(map_init));           axes[0,0].set_title(r"$\mathrm{warcraft}(x_{ini})$")
# axes[0,2].imshow(print_map(map_brls));           axes[0,2].set_title(r"$\mathrm{warcraft}(x^*_{rls})$")
# axes[0,4].imshow(print_map(map_cdsm));           axes[0,4].set_title(r"$\mathrm{warcraft}(x^*_{dsm})$")
# axes[0,6].imshow(print_map(map_atck));           axes[0,6].set_title(r"$\mathrm{warcraft}(x^*_{atk})$")
# axes[0,8].imshow(print_map(map_hybr));           axes[0,8].set_title(r"$\mathrm{warcraft}(x^*_{hyb})$")
# axes[0,1].imshow(print_cost_map(cost_map_init)); axes[0,1].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x_{ini}))$")
# axes[0,3].imshow(print_cost_map(cost_map_brls)); axes[0,3].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{rls}))$")
# axes[0,5].imshow(print_cost_map(cost_map_cdsm)); axes[0,5].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{dsm}))$")
# axes[0,7].imshow(print_cost_map(cost_map_atck)); axes[0,7].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{atk}))$")
# axes[0,9].imshow(print_cost_map(cost_map_hybr)); axes[0,9].set_title(r"$\mathrm{costmap}(\mathrm{warcraft}(x^*_{hyb}))$")
# # Plots of the initial path for each image
# axes[1,0].imshow(print_map_plus_path(map_init, opt_path_init));           axes[1,0].set_title(r"$+ \mathrm{p}^*_{ini}$")
# axes[1,2].imshow(print_map_plus_path(map_brls, opt_path_init));           axes[1,2].set_title(r"$+ \mathrm{p}^*_{ini}$")
# axes[1,4].imshow(print_map_plus_path(map_cdsm, opt_path_init));           axes[1,4].set_title(r"$+ \mathrm{p}^*_{ini}$")
# axes[1,6].imshow(print_map_plus_path(map_atck, opt_path_init));           axes[1,6].set_title(r"$+ \mathrm{p}^*_{ini}$")
# axes[1,8].imshow(print_map_plus_path(map_hybr, opt_path_init));           axes[1,8].set_title(r"$+ \mathrm{p}^*_{ini}$")
# axes[1,1].imshow(print_cost_map_plus_path(cost_map_init, opt_path_init)); axes[1,1].set_title(format_cost_for_title(cost_map_init, opt_path_init, "x_{ini}",   "\mathrm{p}^*_{ini}"))
# axes[1,3].imshow(print_cost_map_plus_path(cost_map_brls, opt_path_init)); axes[1,3].set_title(format_cost_for_title(cost_map_brls, opt_path_init, "x^*_{rls}", "\mathrm{p}^*_{ini}"))
# axes[1,5].imshow(print_cost_map_plus_path(cost_map_cdsm, opt_path_init)); axes[1,5].set_title(format_cost_for_title(cost_map_cdsm, opt_path_init, "x^*_{dsm}", "\mathrm{p}^*_{ini}"))
# axes[1,7].imshow(print_cost_map_plus_path(cost_map_atck, opt_path_init)); axes[1,7].set_title(format_cost_for_title(cost_map_atck, opt_path_init, "x^*_{atk}", "\mathrm{p}^*_{ini}"))
# axes[1,9].imshow(print_cost_map_plus_path(cost_map_hybr, opt_path_init)); axes[1,9].set_title(format_cost_for_title(cost_map_hybr, opt_path_init, "x^*_{hyb}", "\mathrm{p}^*_{ini}"))
# # Plots of the target path on each image
# axes[2,0].imshow(print_map_plus_path(map_init, opt_path_targ)); axes[2,0].set_title(r"$+ \mathrm{p}^\sharp$")
# axes[2,2].imshow(print_map_plus_path(map_brls, opt_path_targ)); axes[2,2].set_title(r"$+ \mathrm{p}^\sharp$")
# axes[2,4].imshow(print_map_plus_path(map_cdsm, opt_path_targ)); axes[2,4].set_title(r"$+ \mathrm{p}^\sharp$")
# axes[2,6].imshow(print_map_plus_path(map_atck, opt_path_targ)); axes[2,6].set_title(r"$+ \mathrm{p}^\sharp$")
# axes[2,8].imshow(print_map_plus_path(map_hybr, opt_path_targ)); axes[2,8].set_title(r"$+ \mathrm{p}^\sharp$")
# axes[2,1].imshow(print_cost_map_plus_path(cost_map_init, opt_path_targ)); axes[2,1].set_title(format_cost_for_title(cost_map_init, opt_path_targ, "x_{ini}",   "\mathrm{p}^\sharp"))
# axes[2,3].imshow(print_cost_map_plus_path(cost_map_brls, opt_path_targ)); axes[2,3].set_title(format_cost_for_title(cost_map_brls, opt_path_targ, "x^*_{rls}", "\mathrm{p}^\sharp"))
# axes[2,5].imshow(print_cost_map_plus_path(cost_map_cdsm, opt_path_targ)); axes[2,5].set_title(format_cost_for_title(cost_map_cdsm, opt_path_targ, "x^*_{dsm}", "\mathrm{p}^\sharp"))
# axes[2,7].imshow(print_cost_map_plus_path(cost_map_atck, opt_path_targ)); axes[2,7].set_title(format_cost_for_title(cost_map_atck, opt_path_targ, "x^*_{atk}", "\mathrm{p}^\sharp"))
# axes[2,9].imshow(print_cost_map_plus_path(cost_map_hybr, opt_path_targ)); axes[2,9].set_title(format_cost_for_title(cost_map_hybr, opt_path_targ, "x^*_{hyb}", "\mathrm{p}^\sharp"))

# for i in range(3):
#     for j in range(10):
#         # axes[i,j].axis("off")
#         # axes[i,j].set_frame_on(True)
#         axes[i,j].set_xticks([])
#         axes[i,j].set_yticks([])

# fig.tight_layout()
# fig.subplots_adjust(wspace=0.20, hspace=0.20, top=0.95, bottom=0.01, left=0.01, right=0.99)

# fig.savefig("/".join([path_root, "results_maps.pdf"]))


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

def format_cost_for_title(cost_map, path, W_name, p_name):
    cost = compute_path_cost(cost_map, path)
    s = r"$\mathrm{costpath}(" + W_name + ", " + p_name + ") = " + f"{cost:.3f}".zfill(7) + "$"
    return s


# Columns = (raw image, cost map, raw image + initial path, cost map + initial path, raw image + target path, cost map + target path)
# Rows = (initial map, hybrid map, attack map, backprop map, cdsm map); bfgs is not included in this image
R = 6
fig, axes = plt.subplots(nrows=R, ncols=6, figsize=(13, 10))
# Plots of the raw images
i = 0
axes[i,0].imshow(print_map(map_init));           axes[i,0].set_title(r"$\mathcal{W}_\mathrm{init} \triangleq \mathrm{warcraft}(x_{init})$"); i += 1
axes[i,0].imshow(print_map(map_hybr));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{hybr} \triangleq \mathrm{warcraft}(x^*_{hybr})$"); i += 1
axes[i,0].imshow(print_map(map_atck));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{atck} \triangleq \mathrm{warcraft}(x^*_{atck})$"); i += 1
axes[i,0].imshow(print_map(map_back));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{back} \triangleq \mathrm{warcraft}(x^*_{back})$"); i += 1
# axes[i,0].imshow(print_map(map_brls));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{brls} \triangleq \mathrm{warcraft}(x^*_{brls})$"); i += 1
axes[i,0].imshow(print_map(map_bfgs));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{bfgs} \triangleq \mathrm{warcraft}(x^*_{bfgs})$"); i += 1
axes[i,0].imshow(print_map(map_cdsm));           axes[i,0].set_title(r"$\mathcal{W}^*_\mathrm{cdsm} \triangleq \mathrm{warcraft}(x^*_{cdsm})$"); i += 1
# Plots of the cost maps
i = 0
axes[i,1].imshow(print_cost_map(cost_map_init)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}_\mathrm{init})$"); i += 1
axes[i,1].imshow(print_cost_map(cost_map_hybr)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{hybr})$"); i += 1
axes[i,1].imshow(print_cost_map(cost_map_atck)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{atck})$"); i += 1
axes[i,1].imshow(print_cost_map(cost_map_back)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{back})$"); i += 1
# axes[i,1].imshow(print_cost_map(cost_map_brls)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{brls})$"); i += 1
axes[i,1].imshow(print_cost_map(cost_map_bfgs)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{bfgs})$"); i += 1
axes[i,1].imshow(print_cost_map(cost_map_cdsm)); axes[i,1].set_title(r"$\mathrm{costmap}(\mathcal{W}^*_\mathrm{cdsm})$"); i += 1
# Plots of the initial path for each image
i = 0
axes[i,2].imshow(print_map_plus_path(map_init, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
axes[i,2].imshow(print_map_plus_path(map_hybr, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
axes[i,2].imshow(print_map_plus_path(map_atck, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
axes[i,2].imshow(print_map_plus_path(map_back, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
# axes[i,2].imshow(print_map_plus_path(map_brls, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
axes[i,2].imshow(print_map_plus_path(map_bfgs, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
axes[i,2].imshow(print_map_plus_path(map_cdsm, opt_path_init));           axes[i,2].set_title(r"$+ \mathrm{p}^*_{init}$"); i += 1
# Plots of the cost map + initial path for each image
i = 0
axes[i,3].imshow(print_cost_map_plus_path(cost_map_init, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_init, opt_path_init, "\mathcal{W}_\mathrm{init}",   "\mathrm{p}^*_{init}")); i += 1
axes[i,3].imshow(print_cost_map_plus_path(cost_map_hybr, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_hybr, opt_path_init, "\mathcal{W}^*_\mathrm{hybr}", "\mathrm{p}^*_{init}")); i += 1
axes[i,3].imshow(print_cost_map_plus_path(cost_map_atck, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_atck, opt_path_init, "\mathcal{W}^*_\mathrm{atck}", "\mathrm{p}^*_{init}")); i += 1
axes[i,3].imshow(print_cost_map_plus_path(cost_map_back, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_back, opt_path_init, "\mathcal{W}^*_\mathrm{back}", "\mathrm{p}^*_{init}")); i += 1
# axes[i,3].imshow(print_cost_map_plus_path(cost_map_brls, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_brls, opt_path_init, "\mathcal{W}^*_\mathrm{brls}", "\mathrm{p}^*_{init}")); i += 1
axes[i,3].imshow(print_cost_map_plus_path(cost_map_bfgs, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_bfgs, opt_path_init, "\mathcal{W}^*_\mathrm{bfgs}", "\mathrm{p}^*_{init}")); i += 1
axes[i,3].imshow(print_cost_map_plus_path(cost_map_cdsm, opt_path_init)); axes[i,3].set_title(format_cost_for_title(cost_map_cdsm, opt_path_init, "\mathcal{W}^*_\mathrm{cdsm}", "\mathrm{p}^*_{init}")); i += 1
# Plots of the target path on each image
i = 0
axes[i,4].imshow(print_map_plus_path(map_init, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
axes[i,4].imshow(print_map_plus_path(map_hybr, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
axes[i,4].imshow(print_map_plus_path(map_atck, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
axes[i,4].imshow(print_map_plus_path(map_back, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
# axes[i,4].imshow(print_map_plus_path(map_brls, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
axes[i,4].imshow(print_map_plus_path(map_bfgs, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
axes[i,4].imshow(print_map_plus_path(map_cdsm, opt_path_targ));           axes[i,4].set_title(r"$+ \mathrm{p}^\sharp$"); i += 1
# Plots of the cost map + target path for each image
i = 0
axes[i,5].imshow(print_cost_map_plus_path(cost_map_init, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_init, opt_path_targ, "\mathcal{W}_\mathrm{init}",   "\mathrm{p}^\sharp")); i += 1
axes[i,5].imshow(print_cost_map_plus_path(cost_map_hybr, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_hybr, opt_path_targ, "\mathcal{W}^*_\mathrm{hybr}", "\mathrm{p}^\sharp")); i += 1
axes[i,5].imshow(print_cost_map_plus_path(cost_map_atck, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_atck, opt_path_targ, "\mathcal{W}^*_\mathrm{atck}", "\mathrm{p}^\sharp")); i += 1
axes[i,5].imshow(print_cost_map_plus_path(cost_map_back, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_back, opt_path_targ, "\mathcal{W}^*_\mathrm{back}", "\mathrm{p}^\sharp")); i += 1
# axes[i,5].imshow(print_cost_map_plus_path(cost_map_brls, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_brls, opt_path_targ, "\mathcal{W}^*_\mathrm{brls}", "\mathrm{p}^\sharp")); i += 1
axes[i,5].imshow(print_cost_map_plus_path(cost_map_bfgs, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_bfgs, opt_path_targ, "\mathcal{W}^*_\mathrm{bfgs}", "\mathrm{p}^\sharp")); i += 1
axes[i,5].imshow(print_cost_map_plus_path(cost_map_cdsm, opt_path_targ)); axes[i,5].set_title(format_cost_for_title(cost_map_cdsm, opt_path_targ, "\mathcal{W}^*_\mathrm{cdsm}", "\mathrm{p}^\sharp")); i += 1

for i in range(R):
    for j in range(6):
        # axes[i,j].axis("off")
        # axes[i,j].set_frame_on(True)
        axes[i,j].set_xticks([])
        axes[i,j].set_yticks([])

fig.tight_layout()
fig.set_size_inches((13, 13))
fig.subplots_adjust(wspace=0.05, hspace=0.20, top=0.97, bottom=0.01, left=0.0, right=0.98)

fig.savefig("/".join([path_root, "results_maps.pdf"]))
