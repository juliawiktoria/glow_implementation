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
import torch.distributions as distrib
import torch.distributions.transforms as transform

from model_parts import ActivationNormalisation, AffineCoupling, InvertedConvolution

class Block(nn.Module):
    def __init__(self):
        super().__init__()

class NormalisingFlow(nn.Module):
    def __init__(self, dimension, flow_block, num_blocks, density):
        super().__init__()
        flows = []
        for i in range (num_blocks):
            for flow in flow_block:
                flows.append(flow(dimension))
        self.flows = nn.ModuleList(flows)
        self.transforms = transform.ComposeTransform(flows)
        self.base_density = density
        self.final_density = distrib.TransformedDistribution(density, self.transforms)
        self.log_det = []

    def forward(self, z):
        self.log_det = []

        # apply series of flows
        for i in range(len(self.flows)):
            self.log_det.append(self.flows[i].log_abs_det_jacobian(z))
            z = self.flows[i](z)
        return z, self.log_det