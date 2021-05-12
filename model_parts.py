# standard
import numpy as np

# pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

import utilities

class InvertedConvolution(nn.Module):
    def __init__(self, num_features, lu_decomp=False):
        super(InvertedConvolution, self).__init__()
        self.num_features = num_features

        w_init = np.random.randn(num_features, num_features)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weights = nn.Parameter(torch.from_numpy(w_init))
        self.lu_decomp = lu_decomp

    def describe(self):
        print('\t\t\t - > Inv Conv with {} num_features'.format(self.num_features))

    def forward(self, x, sldj, reverse=False):
        lower_det_jacobian = torch.slogdet(self.weights)[1] * x.size(2) * x.size(3)

        if reverse:
            # print("\t\t\t\t\t -> inv conv forward pass")
            weights = self.weights
            sldj = sldj + lower_det_jacobian
        else:
            # print("\t\t\t\t\t -> inv conv reverse pass")
            weights = torch.inverse(self.weights.double()).float()
            sldj = sldj - lower_det_jacobian

        weights = weights.view(self.num_features, self.num_features, 1, 1)
        z = F.conv2d(x, weights)

        return z, sldj

class ActivationNormalisation(nn.Module):
    def __init__(self, num_features, scale=1.):
        super(ActivationNormalisation, self).__init__()
        # buffer acts like a module state, not a parameter
        self.register_buffer('is_initialised', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.epsilon = 1e-6

    def describe(self):
        print('\t\t\t - > Act Norm with {} num_features; bias: {}; logs: {}'.format(self.num_features, self.bias.size(), self.logs.size()))
    
    def init_params(self, x):
        if not self.training:
            print('act norm notr training return from init params')
            return

        with torch.no_grad():
            bias = -1 * utilities.mean_over_dimensions(x.clone(), dim=[0, 2, 3], keepdims=True)
            v = utilities.mean_over_dimensions((x.clone() - bias) ** 2, dim=[0, 2, 3], keepdims=True)
            logs = torch.log(self.scale / (torch.sqrt(v) + 1e-6))

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            
            # setting the buffer to be True
            self.is_initialised += 1
    
    def _center(self, x, reverse=False):
        
        if not reverse:
            return x + self.bias
        else:
            return x - self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs

        if not reverse:
            x = x * logs.exp()
        else:
            x = x * logs.mul(-1).exp()
        
        if sldj is not None:
            lower_det_jacobian = logs.sum() * x.size(2) * x.size(3)
            if not reverse:
                sldj = sldj + lower_det_jacobian
            else:
                sldj = sldj - lower_det_jacobian
        
        return x, sldj
    
    def forward(self, x, lower_det_jacobian=None, reverse=False):
        if not self.is_initialised:
            print('act norm not initialised, initialisning params in the forward pass')
            self.init_params(x)
        
        if not reverse:
            x = self._center(x, reverse)
            x, lower_det_jacobian = self._scale(x, lower_det_jacobian, reverse)
        else:
            x, lower_det_jacobian = self._scale(x, lower_det_jacobian, reverse)
            x = self._center(x, reverse)
        
        return x, lower_det_jacobian

class CNN(nn.Module):
    # cnn for affine coupling layer with an extra hidden layer
    def __init__(self, num_features, hid_layers, out_channels):
        super(CNN, self).__init__()
        norm_function = ActivationNormalisation

        self.in_norm = norm_function(num_features)
        self.in_conv = nn.Conv2d(num_features, hid_layers, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_function(hid_layers)
        self.mid_conv = nn.Conv2d(hid_layers, hid_layers, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        # self.mid_norm_2 = norm_function(hid_layers)
        # self.mid_conv_2 = nn.Conv2d(hid_layers, hid_layers, kernel_size=1, padding=0, bias=False)
        # nn.init.normal_(self.mid_conv_2.weight, 0., 0.05)

        self.out_norm = norm_function(hid_layers)
        self.out_conv = nn.Conv2d(hid_layers, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x, _ = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x, _ = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        # x, _ = self.mid_norm_2(x)
        # x = F.relu(x)
        # x = self.mid_conv_2(x)

        x, _ = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x

class AffineCoupling(nn.Module):
    def __init__(self, num_features, hid_layers):
        super(AffineCoupling, self).__init__()
        self.num_features = num_features
        self.cnn = CNN(num_features, hid_layers, num_features * 2)
        self.scale = nn.Parameter(torch.ones(num_features, 1, 1))

    def describe(self):
        print('\t\t\t - > Aff Coupling with {} num_features'.format(self.num_features))
    
    def forward(self, x, lower_det_jacobian, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)
        
        st = self.cnn(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if not reverse:
            x_change = (x_change + t) * s.exp()
            lower_det_jacobian = lower_det_jacobian + s.flatten(1).sum(-1)
        else:
            x_change = x_change * s.mul(-1).exp() - t
            lower_det_jacobian = lower_det_jacobian - s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)

        return x, lower_det_jacobian

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()
    
    def forward(self, x, reverse=False):
        # get input dimensions
        b, c, h, w = x.size()
        if not reverse:
            # squeeze
            x = x.view(b, c, h // 2, 2, w // 2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * 4, h // 2, w // 2)
            # output shape: (b, 4c, h/2, w/2)
        else:
            # unsqueeze
            x = x.view(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(b, c // 4, h * 2, w * 2)
            # output shape: (b, c/4, 2h, 2w)
        return x