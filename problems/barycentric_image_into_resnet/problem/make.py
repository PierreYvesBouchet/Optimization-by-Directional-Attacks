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
path_build_data = "/".join([path_root, "build_data"])
sys.path.append(path_build_data)



#%% Choice of the device to support the NN (hardcoded to CPU for portability)

if   torch.cuda.is_available():         device = "cuda"
elif torch.backends.mps.is_available(): device = "mps"
else:                                   device = "cpu"
device = "cpu"



#%% Import of the ResNet and images

# name of the target image to be found in the build_data folder
image_target_name = "image_target.jpg"

# Paths
path_image_target = "/".join([path_build_data, image_target_name])
path_dataset      = "/".join([path_build_data, "dataset"])
path_file_classes = "/".join([path_build_data, "classes.txt"])

# ResNet import
resnet = models.resnet18(weights=ResNet18_Weights.DEFAULT).eval()
for param in resnet.parameters(): param.requires_grad = False
resnet_preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()

# Function to import an image from its path and transform it into a ResNet-readable tensor
def import_image(image_path): return Image.open(image_path)
def preprocess_image(image):  return resnet_preprocessor(image)

# Calls Resnet on a given preprocessed image
def compute_resnet(preprocessed_image): return torch.softmax(resnet(preprocessed_image.unsqueeze(0)).squeeze(0), dim=0)

# Output of ResNet(target image)
image_target_preprocessed  = preprocess_image(import_image(path_image_target))
image_target_resnet_output = compute_resnet(image_target_preprocessed)

# Makes two lists of respectively all preprocessed version of dataset images, and their name in the dataset
list_preprocessed_images = [image_target_preprocessed]
list_names_images        = [image_target_name]
for f in os.listdir(path_dataset):
    if os.path.isfile(os.path.join(path_dataset, f)) and not(f == "classes.txt"):
        image_path = "/".join([path_dataset, f])
        list_preprocessed_images.append(preprocess_image(import_image(image_path)))
        list_names_images.append(f)

# Makes a list of the names of all classes predicted by ResNet
with open(path_file_classes, "r") as f: resnet_classes = [s.strip() for s in f.readlines()]



#%% Problem parameters, goal function and constraints function

# (n,m) = (Phi inputs dimension, Phi outputs dimension)
n = 100 # must be at most 1000
m = len(resnet_classes)

# Truncates the two lists of images so that they contain the correct number of elements
list_preprocessed_images = list_preprocessed_images[:n]
list_names_images        = list_names_images[:n]


# Goal function to be maximized: -1 * norm(Phi(x) - resnet(image_target))^2
def  f(y, y_0:torch.tensor=image_target_resnet_output):
    return -1 * (torch.linalg.norm(y-y_0).item()) **2

def df(y, y_0:torch.tensor=image_target_resnet_output):
    return -2*(y-y_0)

# Components of the vector y that do not influence f
inactive_subspace_f = tuple([])


# Constraint set F = [-10,10]^n
def c(x): return (x-10.0)*(x+10.0)



# y = Phi(x) and z = ReLU(c(x)), and yz = [y, z]
def f_tilde(yz, m=m):
    y = yz[:m]
    z = yz[m:]
    return (f(y) - torch.linalg.norm(z)**2).item()

def df_tilde(yz, m=m):
    y = yz[:m]
    z = yz[m:]
    return torch.cat((df(y),-2*z), dim=0)



#%% Class generating the NN Phi

class Phi_model(nn.Module):

    def __init__(self):
        super().__init__()
        self.shape_images = [i for i in list_preprocessed_images[0].shape]
        self.resnet = resnet
        self.resnet_classes = resnet_classes
        self.n = n
        self.m = m
        self.inactive_subspace_f = inactive_subspace_f
        self.mean_image_layer = self.make_mean_image_layer(list_preprocessed_images)

    def get_top_k_categories(self, probabilities, k=5):
        top_probas, top_ids = torch.topk(probabilities, k)
        return [(self.resnet_classes[top_ids[i]], top_probas[i].item()) for i in range(k)]

    def make_mean_image_layer(self, list_preprocessed_images):
        dim_vectorized = math.prod(self.shape_images)
        W_size = torch.Size([dim_vectorized, len(list_preprocessed_images)])
        W = torch.zeros(W_size)
        for i,tensor_image in enumerate(list_preprocessed_images):
            W[:,i] = tensor_image.clone().detach().reshape(-1)
        layer = nn.Linear(self.n, dim_vectorized, bias=False)
        layer.weight = nn.Parameter(W)
        return layer

    @torch.jit.export
    def compute_barycentric_image_from_ponderations(self, x):
        barycenter_as_vector = self.mean_image_layer(x)
        if len(x.shape) == 1: barycenter = torch.reshape(barycenter_as_vector, self.shape_images)
        else:                 barycenter = torch.reshape(barycenter_as_vector, [x.shape[0]]+self.shape_images)
        return barycenter

    def forward(self, x):
        s = torch.softmax(x, dim=-1)
        barycenter_as_vector = self.mean_image_layer(s)
        if len(x.shape) == 1:
            barycenter = torch.reshape(barycenter_as_vector, self.shape_images)
            resnet_output = self.resnet(barycenter.unsqueeze(0)).squeeze(0)
        else:
            barycenter = torch.reshape(barycenter_as_vector, [x.shape[0]]+self.shape_images)
            resnet_output = self.resnet(barycenter)
        output = torch.softmax(resnet_output, dim=-1)
        return output



#%% Class generating the NN Phi_tilde

class Phi_tilde_model(nn.Module):

    def __init__(self, Phi):
        super().__init__()
        self.Phi = Phi
        self.n = self.Phi.n
        self.m = self.Phi.m + len(c(torch.zeros(self.Phi.n)))
        self.c = c
        self.relu = nn.ReLU()
        self.inactive_subspace_f = self.Phi.inactive_subspace_f

    def forward(self, x):
        y = self.Phi(x)
        z = self.relu(self.c(x))
        return torch.cat((y,z), dim=-1)



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

# Saves the ResNet output of the reference image
path_save = "/".join([path_root, "image_target_resnet_output.pt"]); torch.save(image_target_resnet_output, path_save)

# Starting point and radius
x_0 = torch.zeros(Phi.n)
r_0 = 1E0

# Minimal/maximal radius for the poll step and the attack step
r_dsm_min = 1E-5
r_atk_min = 1E-6
r_dsm_max = 1E1
r_atk_max = 1E1

# Global solution x_star, since it is known in this problem
x_star = torch.ones(Phi.n)*-10; x_star[0] = 10

# Saving the parameters
parameters = [r_0, r_atk_min, r_atk_max, r_dsm_min, r_dsm_max]
path_save = "/".join([path_root, "x_0.pt"]);        torch.save(x_0, path_save)
path_save = "/".join([path_root, "parameters.pt"]); torch.save(parameters, path_save)
path_save = "/".join([path_root, "x_star.pt"]);     torch.save(x_star, path_save)

# List of points tested in preliminary study of attack algorithms
x_list_attack_analysis = [x_0, x_star]
for _ in range(98):
    x = -10 + 20*torch.rand(Phi.n)
    x_list_attack_analysis.append(x)
path_save = "/".join([path_root, "attack_analysis_points.pt"]); torch.save(x_list_attack_analysis, path_save)



#%% Temp section: plot of some bacyrentric images

run_temp = False

if run_temp:

    import matplotlib.pyplot as plt

    plt.close("all")

    fig, ax = plt.subplots(nrows = 2, ncols = 4)
    fig.set_size_inches(8,4)
    fig.tight_layout()

    X = []
    for i in range(8): X.append(torch.zeros(Phi.n))
    X[0][0] = 1
    X[1][1] = 1
    X[2][2] = 1
    X[3][3] = 1
    X[4][0] = 1/2; X[4][1] = 1/2
    X[5][0] = 1/4; X[5][2] = 3/4
    X[6][0] = 3/4; X[6][3] = 1/4
    X[7][0] = 1/4; X[7][1] = 1/4; X[7][2] = 1/4; X[7][3] = 1/4
    for i in range(8):
        lig = i // 4
        col = i %  4
        I = Phi.compute_barycentric_image_from_ponderations(X[i])
        # predict = torch.softmax(Phi.resnet(I.unsqueeze(0)).squeeze(0), dim=-1)
        I = np.transpose(I.numpy(), [1,2,0])
        ax[lig,col].imshow(I)
        ax[lig,col].set_axis_off()
        ax[lig,col].set_title(X[i][:4].numpy())

    fig.savefig("barycenters.pdf")
