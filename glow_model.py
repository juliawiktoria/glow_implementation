# standard
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_parts import *

# class for building GlowModel, not to be used on its own
class _FlowStep(nn.Module):
    # comprises of three / four transforms
    def __init__(self, in_channels, mid_channels):
        super(_FlowStep, self).__init__()

        print('initialising flow step with {} in channels.'.format(in_channels))

        # define transforms; hardcoded, not a framework for creating own models with different transforms
        self.normalisation = ActivationNormalisation(in_channels)
        self.convolution = InvertedConvolution(in_channels)
        self.flow_transformation = None
        self.coupling = AffineCoupling(in_channels // 2, mid_channels)

    def forward(self, x, log_det_jacobian=None, reverse=False):
        # normal forward pass [ActNorm, 1x1conv, AffCoupling]
        if not reverse:
            x, log_det_jacobian = self.normalisation(x, log_det_jacobian, reverse)
            x, log_det_jacobian = self.convolution(x, log_det_jacobian, reverse)
            # flow transform step
            x, log_det_jacobian = self.coupling(x, log_det_jacobian, reverse)
        # reversed pass [AffCoupling, 1x1conv, ActNorm]
        else:
            x, log_det_jacobian = self.coupling(x, log_det_jacobian, reverse)
            # flow transform step
            x, log_det_jacobian = self.convolution(x, log_det_jacobian, reverse)
            x, log_det_jacobian = self.normalisation(x, log_det_jacobian, reverse)
            
        return x, log_det_jacobian

# class for building GlowModel, not to be used on its own
class _GlowLevel(nn.Module):
    # creates one glow level
    # level comprises of a squeeze step, K flow steps, and split step (except for the last leves, which does not have a split step)
    def __init__(self, in_channels, mid_channels, num_steps, if_split=True):
        super(_GlowLevel, self).__init__()
        # squeeze operation
        self.squeeze = Squeeze()
        # create K steps of the flow K x ([t,t,t]) where t is a flow transform
        # channels are multiplied by 4 to account for squeeze operation that takes place before flow steps
        self.steps = nn.ModuleList([_FlowStep(in_channels=in_channels * 4, mid_channels=mid_channels) for _ in range(num_steps)])
        # split operation is not performed in the last forward level (in the first when reversed)
        self.if_split = if_split
        self.split = Split2d(in_channels * 4)

    def forward(self, x, log_det_jacobian, reverse=False, temp=None):
        # normal forward pass when reverse == False
        if not reverse:
            # print('\t\t\t -> level forward with input size: {}'.format(x.size()))
            # print('input forward: {}'.format(x.size()))
            # 1. squeeze
            x = self.squeeze(x, reverse)
            # print('\t\t\t\t -> after squeeze: {}'.format(x.size()))
            # print('after squeeze: {}'.format(x.size()))
            # 2. apply K flow steps [transform1, transform2, transform3]
            for step in self.steps:
                # print(step.__class__.__name__)
                x, log_det_jacobian = step(x, log_det_jacobian, reverse)
            # print('\t\t\t\t -> after steps: {}'.format(x.size()))

            if self.if_split:
                x, log_det_jacobian = self.split(x, reverse)
                # print('\t\t\t\t -> after split: {}'.format(x.size()))
            # print('after split: {}'.format(x.size()))
        # reverse pass when reverse == True
        else:
            # print('\t\t\t -> level reverse with input size: {}'.format(x.size()))
            if self.if_split:
                x, log_det_jacobian = self.split(x, log_det_jacobian, reverse, temperature=temp)
                # print('\t\t\t\t -> after un-split: {}'.format(x.size()))

            # 2. apply K steps [transform3, transform2, transform1] - reversed order
            for step in reversed(self.steps):
                # print(step.__class__.__name__)
                x, log_det_jacobian = step(x, log_det_jacobian, reverse)
            # print('\t\t\t\t -> after steps: {}'.format(x.size()))

            # 3. un-squeeze
            x = self.squeeze(x, reverse)
            # print('\t\t\t\t -> after un-squeeze: {}'.format(x.size()))
        
        return x, log_det_jacobian

# the whole model
class GlowModel(nn.Module):
    def __init__(self, num_channels, num_levels, num_steps):
        super(GlowModel, self).__init__()
        self.in_channels = 3
        self.num_channels = num_channels
        self.num_levels = num_levels
        self.num_steps = num_steps
        self.in_height = 32
        self.in_width = 32
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        self.out_channels = self.in_channels * (2 ** (num_levels + 1))
        self.out_height = int(self.in_height // (2 ** (num_levels)))
        self.out_width = int(self.in_width // (2 ** (num_levels)))

        self.squeeze = Squeeze()

        self.levels = nn.ModuleList()
        self.create_levels()

    def describe_self(self):
        print('output dimensions of the model: ({}, {}, {}, {})'.format(32, self.out_channels, self.out_height, self.out_width))

    def create_levels(self):
        # creates nn.ModuleList of all levels of the flow`
        # first (L - 1) levels include splitting
        for i in range(self.num_levels - 1):
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
        log_det_jacobian = ldj.flatten(1).sum(-1)
        return y, log_det_jacobian

    def forward(self, x, reverse=False, temp=None):
        # defining first log_det for the forward pass
        if not reverse:
            # print('\t\t-> model forward')
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max [{}, {}]'.format(x.min(), x.max()))
            x, log_det_jacobian = self._pre_process(x)
        # defining first log_det for thereverse pass
        else:
            # print('\t\t-> model reverse') 
            log_det_jacobian = torch.zeros(x.size(0), device=x.device)
            # reverse the ordering of the levels do the no-split level is the first one now
            self.levels = self.levels[::-1]
        # pass the input through all the glow levels iteratively
        # each block solves the direction of the pass within itself
        for level in self.levels:
            x, log_det_jacobian = level(x, log_det_jacobian, reverse, temp)
            # level_number += 1
        # print('after all levels x size:{}\tlog size: {}'.format(x.size(), log_det_jacobian.size()))
        return x, log_det_jacobian