import numpy as np
import torch
import torch.nn.utils as utils
import os

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

# squeezing and unsqueezing (streaching) the model
def squeeze(x, reverse=False):
    b, c, h, w = x.size()

    if reverse:
        # unsqueeze
        x = x.view(b, c // 4, 2, 2, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(b, c // 4, h * 2, w * 2)
    else:
        # squeeze
        x = x.view(b, c, h //2, 2, w //2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(b, c * 2 * 2, h // 2, w // 2)
    
    return x

# clipping gradient norm to avoid exploding gradients
def clip_grad_norm(optimizer, max_norm, norm_type=2):
    for group in optimizer.param_groups:
        utils.clip_grad_norm(group['params'], max_norm, norm_type)

# calculates bpd metric according to the definition
def bits_per_dimension(x, nll):
    dim = np.prod(x.size()[1:])
    bpd = nll / (np.log(2) * dim)
    return bpd

# getting a sample of n (batch_size) images from latent space
def sample(model, batch_size, device):
    z = torch.randn((batch_size, 3, 32, 32), dtype=torch.float32, device=device)
    # use the invertibility principle to get the sample
    x, _ = model(z, reverse=True)
    x = torch.sigmoid(x)

    return x

def save_model_checkpoint(model, epoch, avg_loss, best=False):
    if not best:
        file_name = "checkpoint_epoch_{}.pth.tar".format(epoch)
    else:
        file_name = "best_checkpoint_epoch_{}.pth.tar".format(epoch)
        # saving model in the current epoch to a file
    os.makedir('checkpoints', exist_ok=True)
    saving_path = os.path.join('checkpoins', file_name)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'test_loss': avg_loss}, saving_path)
    print("model saved to a file named {}".format(file_name))