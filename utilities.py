import numpy as np
import torch
import torch.nn.utils as utils
import os
import torch.nn as nn
import torchvision

# ==================== METRIC CALCULATIONS ============================

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

# calculates bpd metric according to the definition
def bits_per_dimension(x, nll):
    b, c, h, w = x.size()
    dim = c * h * w
    bpd = nll / (np.log(2) * dim)
    return bpd

# ====================== INNER CALCS FOR MODEL PARTS AND TRAINING ================

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

# ===================== SAMPLING =============================================

# getting a sample of n (num_samples) images from latent space
@torch.no_grad()
def sample(model, device, args):
    # get a specified number of tensors in the shape of a desired images from the normal random distribution
    z = torch.randn((args.num_samples, args.num_features, args.img_height, args.img_width), dtype=torch.float32, device=device)
    print('z sample size: {}'.format(z.size()))
    # use the invertibility principle to get the sample
    imgs, _ = model(z, reverse=True)
    imgs = torch.sigmoid(imgs)
    return imgs

# ===================== SAVING IMAGES AND CHECKPOINTS ====================

def save_sampled_images(epoch, imgs, num_samples, saving_pth, if_separate=True, if_grid=False):
    grids = 'samples/grids'
    os.makedirs(saving_pth, exist_ok=True) # create a dir for each epoch
    os.makedirs(grids, exist_ok=True) # create a dir for grids
    # save every image separately
    if if_separate:
        for i in range(imgs.size(0)):
            torchvision.utils.save_image(imgs[i, :, :, :], '{}/img_{}.png'.format(saving_pth, i))
    # save images in one grid in one image
    if if_grid:
        # save a grid of images in a pre-made directory, this one is not custom, maybe in the future
        images_concat = torchvision.utils.make_grid(imgs, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        torchvision.utils.save_image(images_concat, '{}/grid_epoch_{}.png'.format(grids, epoch))

def save_model_checkpoint(model, epoch, dataset_name, optimizer, scheduler, avg_loss, best=False):
  # just overwrite a file to know which checkpoint is the best
    if best:
        with open('best_{}_checkpoint.txt'.format(dataset_name), 'w') as file:
          file.write('Epoch with the best loss: {}'.format(epoch))
    # saving model in the current epoch to a file
    file_name = "{}_checkpoint_epoch_{}.pth".format(dataset_name, epoch)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'test_loss': avg_loss,
                'optim': optimizer.state_dict(),
                'sched': scheduler.state_dict()}, file_name)
    print("model saved to a file named {}".format(file_name))