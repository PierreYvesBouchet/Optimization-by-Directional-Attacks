# -*- coding: utf-8 -*-



#%% Libraries import

# Generic Python packages
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

# Torch-related packages
import torch
from torch import nn

# Images-related packages
from PIL import Image
import torchvision.models as models

# Import of the ResNet network
from torchvision.models.resnet import ResNet18_Weights

path_root = os.path.dirname(os.path.abspath(__file__))
path_problem = "/".join([path_root, "..", "problem"])
sys.path.append(path_problem)



#%%

Phi_tilde = torch.jit.load("/".join([path_problem, "Phi_tilde.pt"]))
Phi = Phi_tilde.Phi

result_file_atk = torch.load("/".join([path_root, "optim_attacks.pt"]),       weights_only=True)
result_file_rls = torch.load("/".join([path_root, "optim_line_searches.pt"]), weights_only=True)
result_file_hyb = torch.load("/".join([path_root, "optim_hybrid.pt"]),        weights_only=True)
result_file_dsm = torch.load("/".join([path_root, "optim_dsm.pt"]),           weights_only=True)

list_preprocessed_images = Phi.list_preprocessed_images
x_atk = result_file_atk[-1][0]; s_atk = torch.softmax(x_atk, dim=-1)
x_rls = result_file_rls[-1][0]; s_rls = torch.softmax(x_rls, dim=-1)
x_hyb = result_file_hyb[-1][0]; s_hyb = torch.softmax(x_hyb, dim=-1)
x_dsm = result_file_dsm[-1][0]; s_dsm = torch.softmax(x_dsm, dim=-1)
x_opt = torch.zeros_like(x_atk); x_opt[0] = 10; x_opt[1:] = -10; s_opt = torch.softmax(x_opt, dim=-1)

image_atk = np.transpose(Phi.compute_barycentric_image_from_ponderations(s_atk).numpy(), [1,2,0])
image_rls = np.transpose(Phi.compute_barycentric_image_from_ponderations(s_rls).numpy(), [1,2,0])
image_hyb = np.transpose(Phi.compute_barycentric_image_from_ponderations(s_hyb).numpy(), [1,2,0])
image_dsm = np.transpose(Phi.compute_barycentric_image_from_ponderations(s_dsm).numpy(), [1,2,0])
image_opt = np.transpose(Phi.compute_barycentric_image_from_ponderations(s_opt).numpy(), [1,2,0])

iterates_list = [[image_opt, "image_opt"],
                 [image_rls, "image_rls"],
                 [image_dsm, "image_dsm"],
                 [image_atk, "image_atk"],
                 [image_hyb, "image_hyb"]
                 ]



#%% Temp section: plot of some bacyrentric images

fig, ax = plt.subplots(nrows = 2, ncols = len(iterates_list))
fig.set_size_inches(8,4)
fig.tight_layout()

def delta_image(image): return 1E3*(image-image_opt)

delta_min = min([np.min(delta_image(image)) for image in [i[0] for i in iterates_list]])
delta_max = max([np.max(delta_image(image)) for image in [i[0] for i in iterates_list]])

for i in range(len(iterates_list)):
    image, title = iterates_list[i]
    ax[0,i].imshow(image)
    ax[1,i].imshow(delta_image(image), vmin=delta_min, vmax=delta_max)
    ax[0,i].set_axis_off(); ax[1,i].set_axis_off()
    ax[0,i].set_title("$I = ${:s}".format(title)); ax[1,i].set_title("$10^{3}(I-I_1)$")
    print(delta_image(image))
    print()
