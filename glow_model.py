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
    # creates a chain of levels
    # level comprises of a squeeze step, K flow steps, and split step (except for the last leves, which does not have a split step)
    def __init__(self, num_features, hid_layers, num_steps, num_levels, lvl_id=1):
        super(_GlowLevel, self).__init__()
        # lvl ID for description and reference
        self.lvl_id = lvl_id
        # squeeze operation
        # self.squeeze = Squeeze()
        # create K steps of the flow K x ([t,t,t]) where t is a flow transform
        # channels (features) are multiplied by 4 to account for squeeze operation that takes place before flow steps
        self.flow_steps = nn.ModuleList([_FlowStep(num_features=num_features, hid_layers=hid_layers, step_num=i+1) for i in range(num_steps)])
        self.reversed_steps = reversed(self.flow_steps)

        self.squeeze = Squeeze()

        if num_levels > 1:
            self.next_lvl = _GlowLevel(num_features=num_features*2,
                                       hid_layers=hid_layers,
                                       num_steps=num_steps,
                                       num_levels=num_levels-1,
                                       lvl_id=lvl_id+1)
        else:
            self.next_lvl = None

    def describe(self):
        print('\t - > Level {}'.format(self.lvl_id))
        print('\t\t - > Squeeze layer')
        for step in self.flow_steps:
            step.describe()
        if self.next_lvl is not None:
            print('\t\t - > Split layer')
            self.next_lvl.describe()
    
    def forward(self, x, sum_lower_det_jacobian, reverse=False, temp=None):
        print('lvl #{}, has next: {}'.format(self.lvl_id, self.next_lvl is not None))
        
        # normal forward pass when reverse == False
        if not reverse:
            for step in self.flow_steps:
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)
            
        if self.next_lvl is not None:
            x = self.squeeze(x, reverse)
            x_1, x_2 = x.chunk(2, dim=1)
            x, sum_lower_det_jacobian = self.next_lvl(x_1, sum_lower_det_jacobian, reverse)
            x = torch.cat((x, x_2), dim=1)
            x = self.squeeze(x, reverse=True)
        
        if reverse:
            for step in self.reversed_steps:
                x, sum_lower_det_jacobian = step(x, sum_lower_det_jacobian, reverse)
        
        return x, sum_lower_det_jacobian

# the whole model
class GlowModel(nn.Module):
    def __init__(self, num_features, hid_layers, num_levels, num_steps, img_height, img_width):
        super(GlowModel, self).__init__()
        self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))
        self.num_features = num_features
        self.hid_layers = hid_layers
        self.num_levels = num_levels
        self.num_steps = num_steps

        self.in_height = img_height
        self.in_width = img_width

        self.levels = _GlowLevelRec(num_features=num_features*4,
                                 hid_layers=hid_layers,
                                 num_steps=num_steps,
                                 num_levels=num_levels)

        self.squeeze = Squeeze()

    def describe(self):
        # method for describing the rchitecture of the model, when called it calls all its subparts
        # and produces a nice visualisation of the levels, steps, and transforms
        print('==============GLOW MODEL============')
        self.levels.describe()
        print('====================================')

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
        # defining first logdet for the reversed pass
        else:
            sum_lower_det_jacobian = torch.zeros(x.size(0), device=x.device)
        
        x = self.squeeze(x)
        x, sum_lower_det_jacobian = self.levels(x, sum_lower_det_jacobian, reverse)
        x = self.squeeze(x, reverse=True)

        return x, sum_lower_det_jacobian