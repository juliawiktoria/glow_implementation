## Standard libraries
import os
import math
import time
import numpy as np

## Imports for plotting
import matplotlib.pyplot as plt

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler as sched
import torch.backends.cudnn as cudnn

# Torchvision
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

# other
from tqdm import tqdm

from model import GlowModel
from utilities import AvgMeter, clip_grad_norm, bits_per_dimension, sample, NLLLoss

@torch.enable_grad()
def train(epoch, model, trainloader, device, optimizer, scheduler, loss_func, max_grad_norm):
    # TODO: implement checkpointing
    print("===> EPOCH {}".format(epoch))
    global global_step
    # training mode from torch nn module
    model.train()
    loss_meter = AvgMeter()

    # fancy progress bar for the terminal
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            z, sldj = model(x, reverse=False)
            loss = loss_func(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            loss.backward()

            if max_grad_norm > 0:
                clip_grad_norm(optimizer, max_grad_norm)
            optimizer.step()
            scheduler.step()

            progress_bar.set_postfix(nll=loss_meter.avg, bpd=bits_per_dimension(x, loss_meter.avg), lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))
            global_step += x.size(0)

@torch.no_grad()
def test(epoch, model, testloader, device, loss_func, num_samples):
    print("testing func")
    global best_loss

    model.eval()
    loss_meter = AvgMeter()

    with tqdm(total=len(testloader.dataset)) as progress_bar:
        for x, _ in testloader:
            x = x.to(device)
            z, sldj = model(x, reverse=False)
            loss = loss_func(z, sldj)
            loss_meter.update(loss.item(), x.size(0))
            progress_bar.set_postfix(nll=loss_meter.avg, bpd=bits_per_dimension(x, loss_meter.avg))
            progress_bar.update(x.size(0))

        # Save samples and data
        images = sample(model, num_samples, device) 
        path_to_images = 'samples/epoch' + str(epoch) # custom name for each epoch
        os.makedirs(path_to_images, exist_ok=True) # create a dir for each epoch

        # 
        for i in range(images.size(0)):
            torchvision.utils.save_image(images[i, :, :, :], '{}/img_{}.png'.format(path_to_images, i))
        # images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
        # torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))

def main_wrapper():
    if_gpu = True
    # default args
    batch_size = 64
    benchmark = True
    gpu_ids = [0]
    learning_rate = 1e-3
    max_grad_norm = -1.
    num_channels = 512
    num_levels = 3
    num_steps = 32
    num_epochs = 20
    num_samples = 64
    num_workers = 8
    resume = False
    seed = 0
    warm_up = 500000
    
    device = 'cuda' if torch.cuda.is_available() and if_gpu else 'cpu'
    print(device)
    max_grad_norm_default = -1

    # getting data for training; just CIFAR10
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
            transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)


    # define the model
    model = GlowModel(num_channels, num_levels, num_steps)
    model = model.to(device)
    model.describe()

    # if using GPU
    if device == 'cuda':
        model = torch.nn.DataParallel(model, gpu_ids)

    loss_function = NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / warm_up))

    times_array = []

    # training loop
    print("Starting training of the Glow model")
    for epoch in range(1, num_epochs + 1):
        start_time = time.time()
        train(epoch, model, trainloader, device, optimizer, scheduler, loss_function, max_grad_norm_default)
        test(epoch, model, testloader, device, loss_function, num_samples)
        elapsed_time = time.time() - start_time

        times_array.append(["Epoch " + str(epoch) + ": ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])

    with open("epoch_times.txt", "w") as txt_file:
        for line in times_array:
            txt_file.write(" ".join(line) + "\n")

best_loss = 0
global_loss = 0
main_wrapper()