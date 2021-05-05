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
    def __init__(self, num_features, hid_layers, step_num):
        super(_FlowStep, self).__init__()
        # step ID for description and reference
        self.step_id = step_num

        # define transforms; hardcoded, not a framework for creating own models with different transforms
        self.normalisation = ActivationNormalisation(num_features)
        self.convolution = InvertedConvolution(num_features)
        self.flow_transformation = None
        self.coupling = AffineCoupling(num_features // 2, hid_layers)

    def describe(self):
        print('\t\t - > STEP {}'.format(self.step_id))
        self.normalisation.describe()
        self.convolution.describe()
        self.coupling.describe()

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
    def __init__(self, num_features, hid_layers, num_steps, lvl_num, if_split=True):
        super(_GlowLevel, self).__init__()
        # lvl ID for description and reference
        self.lvl_id = lvl_num
        # squeeze operation
        self.squeeze = Squeeze()
        # create K steps of the flow K x ([t,t,t]) where t is a flow transform
        # channels (features) are multiplied by 4 to account for squeeze operation that takes place before flow steps
        self.steps = nn.ModuleList([_FlowStep(in_channels=num_features * 4, hid_layers=hid_layers, step_num=i+1) for i in range(num_steps)])
        # split operation is not performed in the last forward level (in the first when reversed)
        self.if_split = if_split
        self.split = Split2d(num_features * 4)

    def describe(self):
        print('\t - > Level {}'.format(self.lvl_id))
        print('\t\t - > Squeeze layer')
        for step in self.steps:
            step.describe()
        if self.if_split:
            print('\t\t - > Split layer')
    
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
    def __init__(self, num_features, hid_layers, num_levels, num_steps, img_height, img_width):
        super(GlowModel, self).__init__()
        self.num_features = num_features
        self.hid_layers = hid_layers
        self.num_levels = num_levels
        self.num_steps = num_steps

        # hardcoded for cifar10
        self.in_height = img_height
        self.in_width = img_width

        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

        # computes the size of the output tensor so it can be used to generate input for the reverse model
        self.out_channels = self.num_features * (2 ** (num_levels + 1))
        self.out_height = int(self.in_height // (2 ** (num_levels)))
        self.out_width = int(self.in_width // (2 ** (num_levels)))

        self.levels = nn.ModuleList()
        self.create_levels()

    def describe(self):
        # method for describing the rchitecture of the model, when called it calls all its subparts
        # and produces a nice visualisation of the levels, steps, and transforms
        print('==============GLOW MODEL============')
        for level in self.levels:
            level.describe()
        print('====================================')

    def create_levels(self):
        # creates nn.ModuleList of all levels of the flow`
        # first (L - 1) levels include splitting
        for i in range(self.num_levels - 1):
            self.levels.append(_GlowLevel(in_channels=self.num_features, mid_channels=self.hid_layers, num_steps=self.num_steps, lvl_num=i+1))
            self.in_channels *= 2
        # last layer without the split part
        self.levels.append(_GlowLevel(in_channels=self.num_features, mid_channels=self.hid_layers, num_steps=self.num_steps, lvl_num=self.num_levels, if_split=False))
    
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
            if x.min() < 0 or x.max() > 1:
                raise ValueError('Expected x in [0, 1], got min/max [{}, {}]'.format(x.min(), x.max()))
           
            x, log_det_jacobian = self._pre_process(x)
            # go through all the levels iteratively
            for level in self.levels:
                x, log_det_jacobian = level(x, log_det_jacobian, reverse, temp)
        else:
            # reverse the ordering of the levels do the no-split level is the first one now
            self.reversed_levels = self.levels[::-1]

            # generate log det jacobian for reverse pass, input x should be passed to the function itself from sampling functions
            log_det_jacobian = torch.zeros(x.size(0), device=x.device)
            for level in self.reversed_levels:
                x, log_det_jacobian = level(x, log_det_jacobian, reverse, temp)

        return x, log_det_jacobian