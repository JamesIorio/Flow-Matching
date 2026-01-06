# Basic Imports:
import os # Used for filesystem operations like saving modules or creating folders
import torch # Import PyTorch
import torch.nn as nn # For layers like Conv2d or linear
import torch.nn.functional as F
import torch.optim as optim # For optimizers like Adam or SGD

# Dataset and Visualization Imports:
from torchvision import datasets, transforms # To access MNIST
from torch.utils.data import DataLoader # For resizing, normalizing, and converting images to tensor
from torchvision.utils import make_grid # To put images into a grid
import matplotlib.pyplot as plt # Used for plotting images
from tqdm.auto import tqdm  # For training progress bars

#CFM Imports:
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.models.unet import UNetModel # network architecture to predict velocity field

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True  # Good for fixed-size image generation

print("Using device:", device)
print("Torch version:", torch.__version__)
