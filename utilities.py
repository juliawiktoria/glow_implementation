import numpy as np
import math
import torch
import torch.nn.utils as utils
import os
import torch.nn as nn
import torchvision

# standard meter class used tracking metrics of generative models
class AvgMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    # self-explanatory
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class NLLLoss(nn.Module):
    """Negative log-likelihood loss assuming isotropic gaussian with unit norm.
    Args:
        k (int or float): Number of discrete values in each input dimension.
            E.g., `k` is 256 for natural images.
    See Also:
        Equation (3) in the RealNVP paper: https://arxiv.org/abs/1605.08803
    """
    def __init__(self, k=256):
        super(NLLLoss, self).__init__()
        self.k = k

    def forward(self, z, sldj):
        prior_ll = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior_ll = prior_ll.flatten(1).sum(-1) \
            - np.log(self.k) * np.prod(z.size()[1:])
        ll = prior_ll + sldj
        nll = -ll.mean()
        return nll

# calculates mean of values over every dimension of the tensor if there are multiple dimensions
def mean_over_dimensions(tensor, dim=None, keepdims=False):
    if dim is None:
        return tensor.mean()
    else:
        if isinstance(dim, int):
            dim = [dim]
        dim = sorted(dim)
        for d in dim:
            tensor = tensor.mean(dim=d, keepdim=True)
        if not keepdims:
            for i, d in enumerate(dim):
                tensor.squeeze_(d-i)
        return tensor

# clipping gradient norm to avoid exploding gradients
def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm(group['params'], max_norm, norm_type)

# calculates bpd metric according to the definition
def bits_per_dimension(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)
    return bpd

# getting a sample of n (num_samples) images from latent space
def sample(model, device, args):
    # get a specified number of tensors in the shape of a desired images from the normal random distribution
    c, h, w = model.out_channels, model.out_height, model.out_width
    z = torch.randn((args.num_samples, c, h, w), dtype=torch.float32, device=device)
    # use the invertibility principle to get the sample
    imgs, _ = model(z, reverse=True, temp=1)
    imgs = torch.sigmoid(imgs)
    return imgs

def save_sampled_images(epoch, imgs, num_samples, saving_pth, if_separate=True, if_grid=False):
    os.makedirs(saving_pth, exist_ok=True) # create a dir for each epoch
    # save every image separately
    if if_separate:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i, :, :, :], '{}/img_{}.png'.format(saving_pth, i))
    # save images in one grid in one image
    if if_grid:
      # save a grid of images
            images_concat = torchvision.utils.make_grid(imgs, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
            torchvision.utils.save_image(images_concat, '{}/grid_epoch_{}.png'.format(saving_pth, epoch))

def save_model_checkpoint(model, epoch, optimizer, scheduler, avg_loss, best=False):
  # just overwrite a file to know which checkpoint is the best
    if best:
        with open('best_checkpoint.txt', 'w') as file:
          file.write('Epoch with the best loss: {}'.format(epoch))
    # saving model in the current epoch to a file
    file_name = "checkpoint_epoch_{}.pth".format(epoch)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'test_loss': avg_loss,
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict()}, file_name)
    print("model saved to a file named {}".format(file_name))

# calculation for reverse split layer
def gaussian_sample(mean, logs, temperature=1):
    # Sample from Gaussian with temperature
    z = torch.normal(mean, torch.exp(logs) * temperature)
    return z

def gaussian_p(mean, logs, x):
    """
    lnL = -1/2 * { ln|Var| + ((X - Mu)^T)(Var^-1)(X - Mu) + kln(2*PI) }
            k = 1 (Independent)
            Var = logs ** 2
    """
    c = math.log(2 * math.pi)
    return -0.5 * (logs * 2. + ((x - mean) ** 2) / torch.exp(logs * 2.) + c)

# computing gaussian likelihood
def gaussian_likelihood(mean, logs, x):
    p = gaussian_p(mean, logs, x)
    return torch.sum(p, dim=[1, 2, 3])

def split_feature(tensor, type="split"):
    """
    type = ["split", "cross"]
    """
    C = tensor.size(1)
    if type == "split":
        return tensor[:, :C // 2, ...], tensor[:, C // 2:, ...]
    elif type == "cross":
        return tensor[:, 0::2, ...], tensor[:, 1::2, ...]

def compute_same_pad(kernel_size, stride):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size]

    if isinstance(stride, int):
        stride = [stride]

    assert len(stride) == len(kernel_size),\
        "Pass kernel size and stride both as int, or both as equal length iterable"

    return [((k - 1) * s + 1) // 2 for k, s in zip(kernel_size, stride)]