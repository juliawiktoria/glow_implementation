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

class _FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        self.normalisation = ActivationNormalisation(in_channels, return_lower_det_jacobian=True)
        self.convolution = InvertedConvolution(in_channels)
        self.coupling = AffineCoupling(in_channels // 2, mid_channels)

    def forward(self, x, sldj=None, reverse=False):
        if reverse:
            x, sldj = self.coupling(x, sldj, reverse)
            x, sldj = self.convolution(x, sldj, reverse)
            x, sldj = self.normalisation(x, sldj, reverse)
        else:
            x, sldj = self.normalisation(x, sldj, reverse)
            x, sldj = self.convolution(x, sldj, reverse)
            x, sldj = self.coupling(x, sldj, reverse)
        
        return x, sldj

class _GlowLevel(nn.Module):
    def __init__(self, in_channels, mid_channels, num_levels, num_steps):
        super(_GlowLevel, self).__init__()

        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels, mid_channels=mid_channels) for _ in range(num_steps)])

        # there are more than 1 level; create a link to the next level object
        if num_levels > 1:
            self.next = _GlowLevel(in_channels=2*in_channels, mid_channels=mid_channels, num_levels=num_levels-1, num_steps=num_steps)
        else:
            self.next = None

    def forward(self, x, sldj, reverse=False):
        if not reverse:
            for step in self.steps:
                x, sldj = step(x, sldj, reverse)
        
        if self.next is not None:
            x = squeeze(x)
            x, x_split = x.chunk(2, dim=1)
            x, sldj = self.next(x, sldj, reverse)
            x = torch.cat((x, x_split), dim=1)
            x = squeeze(x, reverse=True)
        
        if reverse:
            for step in reversed(self.steps):
                x, sldj = step(x, sldj, reverse)
        
        return x, sldj

class GlowModel(nn.Module):
    def __init__(self, num_channels, num_layers, num_steps):
        super(GlowModel, self).__init__()
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        self.flows = _GlowLevel(in_channels=4*3, mid_channels=num_channels, num_levels=num_layers, num_steps=num_steps)
    
    def forward(self, x, reverse=False):
        if reverse:
            sldj = torch.zeros(x.size(0), device=x.device)
        else:
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max [{}, {}]'.format(x.min(), x.max()))
            x, sldj = self._pre_process(x)
        
        x = squeeze(x)
        x, sldj = self.flows(x, sldj, reverse)
        x = squeeze(x, reverse=True)

        return x, sldj
    
    def _pre_process(self, x):
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sldj = ldj.flatten(1).sum(-1)

        return y, sldj