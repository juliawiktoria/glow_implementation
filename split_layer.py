# pytorch
import torch
import torch.nn as nn

import utilities

# these classes are used in the iterative glow model, which does not work
# left them for rreference

class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", weight_std=0.05):
        super().__init__()

        if padding == "same":
            padding = utilities.compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding, bias=1)

        # init weight with std
        self.conv.weight.data.normal_(mean=0.0, std=weight_std)

        self.conv.bias.data.zero_()

    def forward(self, input):
        x = self.conv(input)
        return x

# https://github.com/y0ast/Glow-PyTorch/blob/dda8e6762bb025d27f1bb70655bc4dc86f8d619f/modules.py#L273
class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3), stride=(1, 1),
                 padding="same", logscale_factor=3):
        super().__init__()

        if padding == "same":
            padding = utilities.compute_same_pad(kernel_size, stride)
        elif padding == "valid":
            padding = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                              padding)

        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()

        self.logscale_factor = logscale_factor
        self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

    def forward(self, input):
        output = self.conv(input)
        return output * torch.exp(self.logs * self.logscale_factor)

# https://github.com/y0ast/Glow-PyTorch/blob/dda8e6762bb025d27f1bb70655bc4dc86f8d619f/modules.py#L273
class Split2d(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv = Conv2dZeros(num_features // 2, num_features)

    def split2d_prior(self, z):
        h = self.conv(z)
        return utilities.split_feature(h, "cross")

    def forward(self, input, logdet=0., reverse=False, temperature=None):
        if reverse:
            z1 = input
            mean, logs = self.split2d_prior(z1)
            z2 = utilities.gaussian_sample(mean, logs, temperature)
            z = torch.cat((z1, z2), dim=1)
            return z, logdet
        else:
            z1, z2 = utilities.split_feature(input, "split")
            mean, logs = self.split2d_prior(z1)
            logdet = utilities.gaussian_likelihood(mean, logs, z2) + logdet
            return z1, logdet