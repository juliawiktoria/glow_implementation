# standard
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn

from model_parts import ActivationNormalisation, AffineCoupling, InvertedConvolution
from utilities import squeeze

class Block(nn.Module):
    def __init__(self):
        super().__init__()