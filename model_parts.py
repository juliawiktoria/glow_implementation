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

import utilities

class InvertedConvolution(nn.Module):
    def __init__(self, num_channels):
        super(InvertedConvolution, self).__init__()
        self.num_channels = num_channels

        w_init = np.random.randn(num_channels, num_channels)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)
        self.weights = nn.Parameter(torch.from_numpy(w_init))

    def forward(self, x, sldj, reverse=False):
        lower_det_jacobian = torch.slogdet(self.weights)[1] * x.size(2) * x.size(3)

        if reverse:
            weights = torch.inverse(self.weights.double()).float()
            sldj = sldj - lower_det_jacobian
        else:
            weights = self.weights
            sldj = sldj + lower_det_jacobian

        weights = weights.view(self.num_channels, self.num_channels, 1, 1)
        z = F.conv2d(x, weights)

        return z, sldj

class ActivationNormalisation(nn.Module):
    def __init__(self, num_features, scale=1., return_lower_det_jacobian=False):
        super(ActivationNormalisation, self).__init__()
        self.register_buffer('is_initialised', torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.num_features = num_features
        self.scale = float(scale)
        self.epsilon = 1e-6
        self.return_lower_det_jacobian = return_lower_det_jacobian

    def init_params(self, x):
        if not self.training:
            return
        with torch.no_grad():
            bias = -1 * utilities.mean_over_dimensions(x.clone(), dim=[0, 2, 3], keepdims=True)
            print("bias shape: {}".format(bias.size()))
            v = utilities.mean_over_dimensions((x.clone() - bias) ** 2, dim=[0, 2, 3], keepdims=True)
            print("v shape: {}".format(v.size()))
            logs = (self.scale / (v.sqrt() + self.epsilon)).log()

            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_initialised += 1
    
    def _center(self, x, reverse=False):
        if reverse:
            return x - self.bias
        else:
            return x + self.bias

    def _scale(self, x, sldj, reverse=False):
        logs = self.logs

        if reverse:
            x = x * logs.mul(-1).exp()
        else:
            x = x * logs.exp()
        
        if sldj is not None:
            lower_det_jacobian = logs.sum() * x.size(2) * x.size(3)
            if reverse:
                sldj = sldj - lower_det_jacobian
            else:
                sldj = sldj + lower_det_jacobian
        
        return x, sldj
    
    def forward(self, x, lower_det_jacobian=None, reverse=False):
        if not self.is_initialised:
            self.init_params(x)
        
        if reverse:
            x, lower_det_jacobian = self._scale(x, lower_det_jacobian, reverse)
            x = self._center(x, reverse)
        else:
            x = self._center(x, reverse)
            x, lower_det_jacobian = self._scale(x, lower_det_jacobian, reverse)
        
        if self.return_lower_det_jacobian:
            return x, lower_det_jacobian
        
        return x

class CNN(nn.Module):
    # cnn for affine coupling layer with an extra hidden layer
    def __init__(self, in_channels, mid_channels, out_channels):
        super(CNN, self).__init__()
        norm_function = ActivationNormalisation

        self.in_norm = norm_function(in_channels)
        self.in_conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False)
        nn.init.normal_(self.in_conv.weight, 0., 0.05)

        self.mid_norm = norm_function(mid_channels)
        self.mid_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv.weight, 0., 0.05)

        self.mid_norm_2 = norm_function(mid_channels)
        self.mid_conv_2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=1, padding=0, bias=False)
        nn.init.normal_(self.mid_conv_2.weight, 0., 0.05)

        self.out_norm = norm_function(mid_channels)
        self.out_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x):
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.mid_norm(x)
        x = F.relu(x)
        x = self.mid_conv(x)

        x = self.mid_norm_2(x)
        x = F.relu(x)
        x = self.mid_conv_2(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        return x

class AffineCoupling(nn.Module):
    def __init__(self, in_channels, mid_channels):
        super(AffineCoupling, self).__init__()
        self.cnn = CNN(in_channels, mid_channels, in_channels * 2)
        self.scale = nn.Parameter(torch.ones(in_channels, 1, 1))
    
    def forward(self, x, lower_det_jacobian, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        st = self.cnn(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        # Scale and translate
        if reverse:
                x_change = x_change * s.mul(-1).exp() - t
                lower_det_jacobian = lower_det_jacobian - s.flatten(1).sum(-1)
        else:
                x_change = (x_change + t) * s.exp()
                lower_det_jacobian = lower_det_jacobian + s.flatten(1).sum(-1)

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
            x = x.view(b, c, h //2, 2, w //2, 2)
            x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
            x = x.view(b, c * 2 * 2, h // 2, w // 2)
        else:
            # unsqueeze
            x = x.view(b, c // 4, 2, 2, h, w)
            x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
            x = x.view(b, c // 4, h * 2, w * 2)
        return x

class Split(nn.Module):
    def __init__(self, in_channels, if_split=True):
        super(Split, self).__init__()
        self.split = if_split
        # learned prior that can be parametrized which results in better likelihood
        if if_split:
            self.conv_prior = ZeroConv2d(in_channels * 2, in_channels * 4)
        else:
            self.conv_prior = ZeroConv2d(in_channels * 4, in_channels * 8)
    
    def prior(self, x):
        # split or cross tensor
        b, c, h, w = x.size()
        new = self.conv_prior(x)
        return self.split_tensor(new)

    def split_tensor(self, x, type='cross'):
        # tensor splitting
        b, c, h, w = x.size()
        if type == 'cross':
            return x[:, 0::2, ...], x[:, 1::2, ...]
        else:
            return x[:, :c//2, ...], x[:, c//2:, ...]

    def forward(self, x, log_jacobian, reverse=False, eps=None):
        # get input dimensions
        b, c, h, w = x.size()
        # normal forward pass
        if not reverse:
            z_1, z_2 = self.split_tensor(x, type='normal')
            mean, log_sd = self.prior(z_1)
            log_jacobian = utilities.gaussian_likelihood(x, log_sd, mean) + log_jacobian
            return z_1, log_jacobian
        # reverse pass
        else:
            mean, log_sd = self.cross_split_prior(x)
            x_new = utilities.gaussian_sample(mean, log_sd, eps)
            z = torch.cat((x, x_new), dim=1)
            return z, log_jacobian
        
class ZeroConv2d(nn.Module):
    # class for prior in Split
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channels, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out