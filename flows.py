# standard
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg as splin

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn
import torch.distributions as distrib
import torch.distributions.transforms as transform

# my functions
from utilities import mean_over_dimensions



