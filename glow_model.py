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

from model_parts import *

# class for building GlowModel, not to be used on its own
class _FlowStep(nn.Module):
    # comprises of three transforms
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        # define transforms; hardcoded, not a framework for creating own models with different transforms
        self.normalisation = ActivationNormalisation(in_channels, return_lower_det_jacobian=True)
        self.convolution = InvertedConvolution(in_channels)
        self.flow_transformation = None
        self.coupling = AffineCoupling(in_channels // 2, mid_channels)

    def forward(self, x, sum_lower_det_jacobian=None, reverse=False):
        # normal forward pass [ActNorm, 1x1conv, AffCoupling]
        if not reverse:
            x, sum_lower_det_jacobian = self.normalisation(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.convolution(x, sum_lower_det_jacobian, reverse)
            # flow transform step
            x, sum_lower_det_jacobian = self.coupling(x, sum_lower_det_jacobian, reverse)
        # reversed pass [AffCoupling, 1x1conv, ActNorm]
        else:
            x, sum_lower_det_jacobian = self.coupling(x, sum_lower_det_jacobian, reverse)
            # flow transform step
            x, sum_lower_det_jacobian = self.convolution(x, sum_lower_det_jacobian, reverse)
            x, sum_lower_det_jacobian = self.normalisation(x, sum_lower_det_jacobian, reverse)
            
        return x, sum_lower_det_jacobian

# class for building GlowModel, not to be used on its own
class _GlowLevel(nn.Module):
    # creates one glow level
    # level comprises of a squeeze step, K flow steps, and split step (except for the last leves, which does not have a split step)
    def __init__(self, in_channels, mid_channels, num_steps, if_split=True):
        super(_GlowLevel, self).__init__()
        # squeeze operation
        self.squeeze = Squeeze()
        # split operation
        self.split = Split(in_channels, if_split)
        # create K steps of the flow K x ([t,t,t]) where t is a flow transform
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels, mid_channels=mid_channels) for _ in range(num_steps)])

    def forward(self, x, sum_lower_det_jacobian, reverse=False):
        # normal forward pass when reverse == False
        if not reverse:
            # 1. squeeze
            x = self.squeeze(x, reverse)

            # 2. apply K flow steps [transform1, transform2, transform3]
            for step in self.steps:
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)

            # 3. split
            if self.split:
                x, sum_lower_det_jacobian = self.split(x, sum_lower_det_jacobian, reverse)
        # reverse pass when reverse == True
        else:
            # 1. split
            if self.split:
                print('split')

            # 2. apply K steps [transform3, transform2, transform1] - reversed order
            for step in reversed(self.steps):
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)

            # 3. un-squeeze
            x = self.squeeze(x, reverse)
        
        return x, sum_lower_det_jacobian

# the whole model
class GlowModel(nn.Module):
    def __init__(self, in_channels, num_channels, num_layers, num_steps):
        super(GlowModel, self).__init__()
        self.in_channels = in_channels
        self.num_channels = num_channels
        self.num_layers = num_layers
        self.num_steps = num_steps
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        self.squeeze = Squeeze()

        self.levels = nn.ModuleList()
        self.create_levels()

    def create_levels(self):
        # creates nn.ModuleList of all levels of the flow
        # first (L - 1) levels include splitting
        for i in range(self.num_layers - 1):
            self.levels.append(_GlowLevel(in_channels=self.in_channels, mid_channels=self.num_channels, num_steps=self.num_steps))
            self.in_channels *= 2
        # last layer without the split part
        self.levels.append(_GlowLevel(in_channels=self.in_channels, mid_channels=self.num_channels, num_steps=self.num_steps, if_split=False))
    
    def _pre_process(self, x):
        # pre-process input
        y = (x * 255. + torch.rand_like(x)) / 256.
        y = (2 * y - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1. - y).log()

        # Save log-determinant of Jacobian of initial transform
        ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - self.bounds).log() - self.bounds.log())
        sum_lower_det_jacobian = ldj.flatten(1).sum(-1)
        return y, sum_lower_det_jacobian

    def forward(self, x, reverse=False):
        # defining first log_det for the forward pass
        if not reverse:
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max [{}, {}]'.format(x.min(), x.max()))
            x, sum_lower_det_jacobian = self._pre_process(x)
        # defining first log_det for thereverse pass
        else:    
            sum_lower_det_jacobian = torch.zeros(x.size(0), device=x.device)
        
        # x = self.squeeze(x)
        # pass the input through all the glow levels iteratively
        # each block solves the direction of the pass within itself
        for level in self.levels:
            x, sum_lower_det_jacobian = level(x, sum_lower_det_jacobian, reverse)
        x = self.squeeze(x, reverse=True)

        return x, sum_lower_det_jacobian