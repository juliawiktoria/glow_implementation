# # pytorch
# import torch
# import torch.nn as nn

# import utilities

# # these classes are used in the iterative glow model, which does not work
# # left them for rreference

# class Conv2d(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=(3, 3), stride=(1, 1),
#                  padding="same", weight_std=0.05):
#         super().__init__()

#         if padding == "same":
#             padding = utilities.compute_same_pad(kernel_size, stride)
#         elif padding == "valid":
#             padding = 0

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
#                               padding, bias=1)

#         # init weight with std
#         self.conv.weight.data.normal_(mean=0.0, std=weight_std)

#         self.conv.bias.data.zero_()

#     def forward(self, input):
#         x = self.conv(input)
#         return x

# # https://github.com/y0ast/Glow-PyTorch/blob/dda8e6762bb025d27f1bb70655bc4dc86f8d619f/modules.py#L273
# class Conv2dZeros(nn.Module):
#     def __init__(self, in_channels, out_channels,
#                  kernel_size=(3, 3), stride=(1, 1),
#                  padding="same", logscale_factor=3):
#         super().__init__()

#         if padding == "same":
#             padding = utilities.compute_same_pad(kernel_size, stride)
#         elif padding == "valid":
#             padding = 0

#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
#                               padding)

#         self.conv.weight.data.zero_()
#         self.conv.bias.data.zero_()

#         self.logscale_factor = logscale_factor
#         self.logs = nn.Parameter(torch.zeros(out_channels, 1, 1))

#     def forward(self, input):
#         output = self.conv(input)
#         return output * torch.exp(self.logs * self.logscale_factor)

# # https://github.com/y0ast/Glow-PyTorch/blob/dda8e6762bb025d27f1bb70655bc4dc86f8d619f/modules.py#L273
# class Split2d(nn.Module):
#     def __init__(self, num_features):
#         super().__init__()
#         self.conv = Conv2dZeros(num_features // 2, num_features)

#     def split2d_prior(self, z):
#         h = self.conv(z)
#         return utilities.split_feature(h, "cross")

#     def forward(self, input, logdet=0., reverse=False, temperature=None):
#         if reverse:
#             z1 = input
#             mean, logs = self.split2d_prior(z1)
#             z2 = utilities.gaussian_sample(mean, logs, temperature)
#             z = torch.cat((z1, z2), dim=1)
#             return z, logdet
#         else:
#             z1, z2 = utilities.split_feature(input, "split")
#             mean, logs = self.split2d_prior(z1)
#             logdet = utilities.gaussian_likelihood(mean, logs, z2) + logdet
#             return z1, logdet

# # standard
# import numpy as np
# from scipy import linalg as splin

# # pytorch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
# import torch.optim as optim
# import torch.optim.lr_scheduler as sched
# import torch.backends.cudnn as cudnn
# import torch.distributions as distrib
# import torch.distributions.transforms as transform

# # my functions
# from utilities import mean_over_dimensions

# class Flow(transform.Transform, nn.Module):
    
#     def __init__(self):
#         transform.Transform.__init__(self)
#         nn.Module.__init__(self)
    
#     # Init all parameters
#     def init_parameters(self):
#         for param in self.parameters():
#             param.data.uniform_(-0.01, 0.01)
            
#     # Hacky hash bypass
#     def __hash__(self):
#         return nn.Module.__hash__(self)

# # domain and codomain are added as own parameters to each class because I was getting an error and that fixed it
# # https://discuss.pytorch.org/t/solved-pytorch1-8-attributeerror-tanhbijector-object-has-no-attribute-domain/116092

# class PlanarFlow(Flow):
#     def __init__(self, dim, h=torch.tanh, hp=(lambda x: 1 - torch.tanh(x) ** 2)):
#         super(PlanarFlow, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(1, dim))
#         self.scale = nn.Parameter(torch.Tensor(1, dim))
#         self.bias = nn.Parameter(torch.Tensor(1))
#         self.h = h
#         self.hp = hp
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
#         self.init_parameters()

#     def _call(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         return z + self.scale * self.h(f_z)

#     def log_abs_det_jacobian(self, z):
#         f_z = F.linear(z, self.weight, self.bias)
#         psi = self.hp(f_z) * self.weight
#         det_grad = 1 + torch.mm(psi, self.scale.t())
#         return torch.log(det_grad.abs() + 1e-9)

# class RadialFlow(Flow):

#     def __init__(self, dim):
#         super(RadialFlow, self).__init__()
#         self.z0 = nn.Parameter(torch.Tensor(1, dim))
#         self.alpha = nn.Parameter(torch.Tensor(1))
#         self.beta = nn.Parameter(torch.Tensor(1))
#         self.dim = dim
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
#         self.init_parameters()

#     def _call(self, z):
#         r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
#         h = 1 / (self.alpha + r)
#         return z + (self.beta * h * (z - self.z0))

#     def log_abs_det_jacobian(self, z):
#         r = torch.norm(z - self.z0, dim=1).unsqueeze(1)
#         h = 1 / (self.alpha + r)
#         hp = - 1 / (self.alpha + r) ** 2
#         bh = self.beta * h
#         det_grad = ((1 + bh) ** self.dim - 1) * (1 + bh + self.beta * hp * r)
#         return torch.log(det_grad.abs() + 1e-9)

# class PReLUFlow(Flow):
#     def __init__(self, dim):
#         super(PReLUFlow, self).__init__()
#         self.alpha = nn.Parameter(torch.Tensor([1]))
#         self.bijective = True
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()

#     def init_parameters(self):
#         for param in self.parameters():
#             param.data.uniform_(0.01, 0.99)

#     def _call(self, z):
#         return torch.where(z >= 0, z, torch.abs(self.alpha) * z)

#     def _inverse(self, z):
#         return torch.where(z >= 0, z, torch.abs(1. / self.alpha) * z)

#     def log_abs_det_jacobian(self, z):
#         I = torch.ones_like(z)
#         J = torch.where(z >= 0, I, self.alpha * I)
#         log_abs_det = torch.log(torch.abs(J) + 1e-5)
#         return torch.sum(log_abs_det, dim = 1)



# class BatchNormFlow(Flow):

#     def __init__(self, dim, momentum=0.95, eps=1e-5):
#         super(BatchNormFlow, self).__init__()
#         # Running batch statistics
#         self.r_mean = torch.zeros(dim)
#         self.r_var = torch.ones(dim)
#         # Momentum
#         self.momentum = momentum
#         self.eps = eps
#         # Trainable scale and shift (cf. original paper)
#         self.gamma = nn.Parameter(torch.ones(dim))
#         self.beta = nn.Parameter(torch.zeros(dim))
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
        
#     def _call(self, z):
#         if self.training:
#             # Current batch stats
#             self.b_mean = z.mean(0)
#             self.b_var = (z - self.b_mean).pow(2).mean(0) + self.eps
#             # Running mean and var
#             self.r_mean = self.momentum * self.r_mean + ((1 - self.momentum) * self.b_mean)
#             self.r_var = self.momentum * self.r_var + ((1 - self.momentum) * self.b_var)
#             mean = self.b_mean
#             var = self.b_var
#         else:
#             mean = self.r_mean
#             var = self.r_var
#         x_hat = (z - mean) / var.sqrt()
#         y = self.gamma * x_hat + self.beta
#         return y

#     def _inverse(self, x):
#         if self.training:
#             mean = self.b_mean
#             var = self.b_var
#         else:
#             mean = self.r_mean
#             var = self.r_var
#         x_hat = (x - self.beta) / self.gamma
#         y = x_hat * var.sqrt() + mean
#         return y
        
#     def log_abs_det_jacobian(self, z):
#         # Here we only need the variance
#         mean = z.mean(0)
#         var = (z - mean).pow(2).mean(0) + self.eps
#         log_det = torch.log(self.gamma) - 0.5 * torch.log(var + self.eps)
#         return torch.sum(log_det, -1)

# class AffineFlow(Flow):
#     def __init__(self, dim):
#         super(AffineFlow, self).__init__()
#         self.weights = nn.Parameter(torch.Tensor(dim, dim))
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
#         nn.init.orthogonal_(self.weights)

#     def _call(self, z):
#         return z @ self.weights
    
#     def _inverse(self, z):
#         return z @ torch.inverse(self.weights)

#     def log_abs_det_jacobian(self, z):
#         return torch.slogdet(self.weights)[-1].unsqueeze(0).repeat(z.size(0), 1)

# class AffineLUFlow(Flow):
#     def __init__(self, dim):
#         super(AffineLUFlow, self).__init__()
#         weights = torch.Tensor(dim, dim)
#         nn.init.orthogonal_(weights)
#         # Compute the parametrization
#         P, L, U = splin.lu(weights.numpy())
#         self.P = torch.Tensor(P)
#         self.L = nn.Parameter(torch.Tensor(L))
#         self.U = nn.Parameter(torch.Tensor(U))
#         # Need to create masks for enforcing triangular matrices
#         self.mask_low = torch.tril(torch.ones(weights.size()), -1)
#         self.mask_up = torch.triu(torch.ones(weights.size()), -1)
#         self.I = torch.eye(weights.size(0))
#         # Now compute s
#         self.s = nn.Parameter(torch.Tensor(np.diag(U)))
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()

#     def _call(self, z):
#         L = self.L * self.mask_low + self.I
#         U = self.U * self.mask_up + torch.diag(self.s)
#         weights = self.P @ L @ U
#         return z @ weights
    
#     def _inverse(self, z):
#         L = self.L * self.mask_low + self.I
#         U = self.U * self.mask_up + torch.diag(self.s)
#         weights = self.P @ L @ U
#         return z @ torch.inverse(self.weights)

#     def log_abs_det_jacobian(self, z):
#         return torch.sum(torch.log(torch.abs(self.s))).unsqueeze(0).repeat(z.size(0), 1)


# class AffineCouplingFlow(Flow):
#     def __init__(self, dim, n_hidden=64, n_layers=3, activation=nn.ReLU):
#         super(AffineCouplingFlow, self).__init__()
#         self.k = dim // 2
#         self.g_mu = self.transform_net(self.k, dim - self.k, n_hidden, n_layers, activation)
#         self.g_sig = self.transform_net(self.k, dim - self.k, n_hidden, n_layers, activation)
#         self.bijective = True
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
#         self.init_parameters()

#     def transform_net(self, nin, nout, nhidden, nlayer, activation):
#         net = nn.ModuleList()
#         for l in range(nlayer):
#             net.append(nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden))
#             net.append(activation())
#         return nn.Sequential(*net)
        
#     def _call(self, z):
#         z_k, z_D = z[:, :self.k], z[:, self.k:]
#         zp_D = z_D * torch.exp(self.g_sig(z_k)) + self.g_mu(z_k)
#         return torch.cat((z_k, zp_D), dim = 1)

#     def _inverse(self, z):
#         zp_k, zp_D = z[:, :self.k], z[:, self.k:]
#         z_D = (zp_D - self.g_mu(zp_k)) / self.g_sig(zp_k)
#         return torch.cat((zp_k, z_D))

#     def log_abs_det_jacobian(self, z):
#         z_k = z[:, :self.k]
#         return -torch.sum(torch.abs(self.g_sig(z_k)))

# class ReverseFlow(Flow):

#     def __init__(self, dim):
#         super(ReverseFlow, self).__init__()
#         self.permute = torch.arange(dim-1, -1, -1)
#         self.inverse = torch.argsort(self.permute)
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
        
#     def _call(self, z):
#         return z[:, self.permute]

#     def _inverse(self, z):
#         return z[:, self.inverse]
    
#     def log_abs_det_jacobian(self, z):
#         return torch.zeros(z.shape[0], 1)


# class ShuffleFlow(ReverseFlow):

#     def __init__(self, dim):
#         super(ShuffleFlow, self).__init__(dim)
#         self.permute = torch.randperm(dim)
#         self.inverse = torch.argsort(self.permute)
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()

# class MaskedCouplingFlow(Flow):
#     def __init__(self, dim, mask=None, n_hidden=64, n_layers=2, activation=nn.ReLU):
#         super(MaskedCouplingFlow, self).__init__()
#         self.k = dim // 2
#         self.g_mu = self.transform_net(dim, dim, n_hidden, n_layers, activation)
#         self.g_sig = self.transform_net(dim, dim, n_hidden, n_layers, activation)
#         self.mask = mask or torch.cat((torch.ones(self.k), torch.zeros(self.k))).detach()
#         self.bijective = True
#         self.domain = torch.distributions.constraints.Constraint()
#         self.codomain = torch.distributions.constraints.Constraint()
#         self.init_parameters()

#     def transform_net(self, nin, nout, nhidden, nlayer, activation):
#         net = nn.ModuleList()
#         for l in range(nlayer):
#             module = nn.Linear(l==0 and nin or nhidden, l==nlayer-1 and nout or nhidden)
#             module.weight.data.uniform_(-0.1, 0.1)
#             net.append(module)
#             net.append(activation())
#         return nn.Sequential(*net)
        
#     def _call(self, z):
#         z_k = (self.mask * z)
#         zp_D = z * torch.exp(self.g_sig(z_k)) + self.g_mu(z_k)
#         return z_k + (1 - self.mask) * zp_D

#     def _inverse(self, z):
#         zp_k = (self.mask * z)
#         z_D = (((1 - self.mask) * z) - self.g_mu(zp_k)) / self.g_sig(zp_k)
#         return zp_k + z_D

#     def log_abs_det_jacobian(self, z):
#         return -torch.sum(torch.abs(self.g_sig(z * self.mask)))

# # standard
# import numpy as np

# # pytorch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from model_parts import *

# # class for building GlowModel, not to be used on its own
# class _FlowStep(nn.Module):
#     # comprises of three / four transforms
#     def __init__(self, num_features, hid_layers, step_num):
#         super(_FlowStep, self).__init__()
#         # step ID for description and reference
#         self.step_id = step_num

#         # define transforms; hardcoded, not a framework for creating own models with different transforms
#         self.normalisation = ActivationNormalisation(num_features)
#         self.convolution = InvertedConvolution(num_features)
#         self.flow_transformation = None
#         self.coupling = AffineCoupling(num_features // 2, hid_layers)

#     def describe(self):
#         print('\t\t - > STEP {}'.format(self.step_id))
#         self.normalisation.describe()
#         self.convolution.describe()
#         self.coupling.describe()

#     def forward(self, x, log_det_jacobian=None, reverse=False):
#         # normal forward pass [ActNorm, 1x1conv, AffCoupling]
#         if not reverse:
#             x, log_det_jacobian = self.normalisation(x, log_det_jacobian, reverse)
#             x, log_det_jacobian = self.convolution(x, log_det_jacobian, reverse)
#             # flow transform step
#             x, log_det_jacobian = self.coupling(x, log_det_jacobian, reverse)
#         # reversed pass [AffCoupling, 1x1conv, ActNorm]
#         else:
#             x, log_det_jacobian = self.coupling(x, log_det_jacobian, reverse)
#             # flow transform step
#             x, log_det_jacobian = self.convolution(x, log_det_jacobian, reverse)
#             x, log_det_jacobian = self.normalisation(x, log_det_jacobian, reverse)
            
#         return x, log_det_jacobian

# # class for building GlowModel, not to be used on its own
# class _GlowLevel(nn.Module):
#     # creates one glow level
#     # level comprises of a squeeze step, K flow steps, and split step (except for the last leves, which does not have a split step)
#     def __init__(self, num_features, hid_layers, num_steps, lvl_num, if_split=True):
#         super(_GlowLevel, self).__init__()
#         # lvl ID for description and reference
#         self.lvl_id = lvl_num
#         # squeeze operation
#         self.squeeze = Squeeze()
#         # create K steps of the flow K x ([t,t,t]) where t is a flow transform
#         # channels (features) are multiplied by 4 to account for squeeze operation that takes place before flow steps
#         self.steps = nn.ModuleList([_FlowStep(num_features=num_features * 4, hid_layers=hid_layers, step_num=i+1) for i in range(num_steps)])
#         # split operation is not performed in the last forward level (in the first when reversed)
#         self.if_split = if_split
#         self.split = Split2d(num_features * 4)

#     def describe(self):
#         print('\t - > Level {}'.format(self.lvl_id))
#         print('\t\t - > Squeeze layer')
#         for step in self.steps:
#             step.describe()
#         if self.if_split:
#             print('\t\t - > Split layer')
    
#     def forward(self, x, log_det_jacobian, reverse=False, temp=None):
#         # normal forward pass when reverse == False
#         if not reverse:
#             # print('\t\t\t -> level forward with input size: {}'.format(x.size()))
#             # print('input forward: {}'.format(x.size()))
#             # 1. squeeze
#             x = self.squeeze(x, reverse)
#             # print('\t\t\t\t -> after squeeze: {}'.format(x.size()))
#             # print('after squeeze: {}'.format(x.size()))
#             # 2. apply K flow steps [transform1, transform2, transform3]
#             for step in self.steps:
#                 # print(step.__class__.__name__)
#                 x, log_det_jacobian = step(x, log_det_jacobian, reverse)
#             # print('\t\t\t\t -> after steps: {}'.format(x.size()))

#             if self.if_split:
#                 x, log_det_jacobian = self.split(x, reverse)
#                 # print('\t\t\t\t -> after split: {}'.format(x.size()))
#             # print('after split: {}'.format(x.size()))
#         # reverse pass when reverse == True
#         else:
#             # print('\t\t\t -> level reverse with input size: {}'.format(x.size()))
#             if self.if_split:
#                 x, log_det_jacobian = self.split(x, log_det_jacobian, reverse, temperature=temp)
#                 # print('\t\t\t\t -> after un-split: {}'.format(x.size()))

#             # 2. apply K steps [transform3, transform2, transform1] - reversed order
#             for step in reversed(self.steps):
#                 # print(step.__class__.__name__)
#                 x, log_det_jacobian = step(x, log_det_jacobian, reverse)
#             # print('\t\t\t\t -> after steps: {}'.format(x.size()))

#             # 3. un-squeeze
#             x = self.squeeze(x, reverse)
#             # print('\t\t\t\t -> after un-squeeze: {}'.format(x.size()))
        
#         return x, log_det_jacobian

# # the whole model
# class GlowModel(nn.Module):
#     def __init__(self, num_features, hid_layers, num_levels, num_steps, img_height, img_width):
#         super(GlowModel, self).__init__()
#         self.num_features = num_features
#         self.hid_layers = hid_layers
#         self.num_levels = num_levels
#         self.num_steps = num_steps

#         # hardcoded for cifar10
#         self.in_height = img_height
#         self.in_width = img_width

#         self.register_buffer('bounds', torch.tensor([0.9], dtype=torch.float32))

#         # computes the size of the output tensor so it can be used to generate input for the reverse model
#         self.out_features = self.num_features * (2 ** (num_levels + 1))
#         self.out_height = int(self.in_height // (2 ** (num_levels)))
#         self.out_width = int(self.in_width // (2 ** (num_levels)))

#         self.levels = nn.ModuleList()
#         self.create_levels()

#     def describe(self):
#         # method for describing the rchitecture of the model, when called it calls all its subparts
#         # and produces a nice visualisation of the levels, steps, and transforms
#         print('==============GLOW MODEL============')
#         for level in self.levels:
#             level.describe()
#         print('====================================')

#     def create_levels(self):
#         # creates nn.ModuleList of all levels of the flow`
#         # first (L - 1) levels include splitting
#         for i in range(self.num_levels - 1):
#             self.levels.append(_GlowLevel(num_features=self.num_features, hid_layers=self.hid_layers, num_steps=self.num_steps, lvl_num=i+1))
#             self.num_features *= 2
#         # last layer without the split part
#         self.levels.append(_GlowLevel(num_features=self.num_features, hid_layers=self.hid_layers, num_steps=self.num_steps, lvl_num=self.num_levels, if_split=False))
    
#     def _pre_process(self, x):
#         # pre-process input
#         y = (x * 255. + torch.rand_like(x)) / 256.
#         y = (2 * y - 1) * self.bounds
#         y = (y + 1) / 2
#         y = y.log() - (1. - y).log()

#         # Save log-determinant of Jacobian of initial transform
#         ldj = F.softplus(y) + F.softplus(-y) - F.softplus((1. - self.bounds).log() - self.bounds.log())
#         log_det_jacobian = ldj.flatten(1).sum(-1)
#         return y, log_det_jacobian

#     def forward(self, x, reverse=False, temp=None):
#         # defining first log_det for the forward pass
#         if not reverse:
#             if x.min() < 0 or x.max() > 1:
#                 raise ValueError('Expected x in [0, 1], got min/max [{}, {}]'.format(x.min(), x.max()))
           
#             x, log_det_jacobian = self._pre_process(x)
#             # go through all the levels iteratively
#             for level in self.levels:
#                 x, log_det_jacobian = level(x, log_det_jacobian, reverse, temp)
#         else:
#             # reverse the ordering of the levels do the no-split level is the first one now
#             self.reversed_levels = self.levels[::-1]

#             # generate log det jacobian for reverse pass, input x should be passed to the function itself from sampling functions
#             log_det_jacobian = torch.zeros(x.size(0), device=x.device)
#             for level in self.reversed_levels:
#                 x, log_det_jacobian = level(x, log_det_jacobian, reverse, temp)

#         return x, log_det_jacobian

# standard
# import numpy as np

# # pytorch
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as data
# import torch.optim as optim
# import torch.optim.lr_scheduler as sched
# import torch.backends.cudnn as cudnn
# import torch.distributions as distrib
# import torch.distributions.transforms as transform

# from model_parts import ActivationNormalisation, AffineCoupling, InvertedConvolution

# class Block(nn.Module):
#     def __init__(self):
#         super().__init__()

# class NormalisingFlow(nn.Module):
#     def __init__(self, dimension, flow_block, num_blocks, density):
#         super().__init__()
#         flows = []
#         for i in range (num_blocks):
#             for flow in flow_block:
#                 flows.append(flow(dimension))
#         self.flows = nn.ModuleList(flows)
#         self.transforms = transform.ComposeTransform(flows)
#         self.base_density = density
#         self.final_density = distrib.TransformedDistribution(density, self.transforms)
#         self.log_det = []

#     def forward(self, z):
#         self.log_det = []

#         # apply series of flows
#         for i in range(len(self.flows)):
#             self.log_det.append(self.flows[i].log_abs_det_jacobian(z))
#             z = self.flows[i](z)
#         return z, self.log_det