## Standard libraries
import os
import math
import time
import numpy as np
import argparse

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

from model import GlowModel
from utilities import *

@torch.enable_grad()
def train(epoch, model, trainloader, device, optimizer, scheduler, loss_func, max_grad_norm):
    # TODO: implement checkpointing
    print("===> EPOCH {}".format(epoch))
    global global_step
    # training mode from torch nn module
    model.train()
    loss_meter = AvgMeter()

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

        global_step += x.size(0)

@torch.no_grad()
def test(epoch, model, testloader, device, loss_func, num_samples, args):
    print("testing func")
    global best_loss

    model.eval()
    loss_meter = AvgMeter()

    for x, _ in testloader:
        x = x.to(device)
        z, sldj = model(x, reverse=False)
        loss = loss_func(z, sldj)
        loss_meter.update(loss.item(), x.size(0))

    if epoch % args.ckpt_interval:
        print('Saving checkpoint file from the epoch #{}'.format(epoch))

    # Save samples and data on the specified interval
    if epoch % args.img_interval:
        print("saving images from the epoch #{}".format(epoch))
        images = sample(model, num_samples, device) 
        path_to_images = 'samples/epoch' + str(epoch) # custom name for each epoch
        os.makedirs(path_to_images, exist_ok=True) # create a dir for each epoch

        for i in range(images.size(0)):
            torchvision.utils.save_image(images[i, :, :, :], '{}/img_{}.png'.format(path_to_images, i))
    # images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
    # torchvision.utils.save_image(images_concat, 'samples/epoch_{}.png'.format(epoch))

def main_wrapper():
    # parsing args for easier running of the program
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', default=False, help='Flag indicating GPU use.')
    parser.add_argument('-num_channels', type=int, default=512, help='Number of channels.')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of flow levels.')
    parser.add_argument('--num_steps', type=int, default=32, help='Number of flow steps.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--resume_training', action='store_true', default=False, help='Flag indicating resuming training from checkpoint.')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for datasets.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer.')
    parser.add_argument('--grad_norm', type=float, default=-1, help="Maximum value of gradient.")
    parser.add_argument('-sched_warmup', type=int, default=500000, help='Warm-up period for scheduler.')
    parser.add_argument('--ckpt_interval', type=int, default=1, help='Create a checkpoint file every N epochs.')
    parser.add_argument('--img_interval', type=int, default=1, help='Generate images every N epochs.')
    parser.add_argument('--ckpt_path', type=str, default='NONE', help='Path to the checkpoint file to use.')
    parser.add_argument('--expr_id', type=str, default='1', help='Experiment ID for logging and identification.')

    args = parser.parse_args()
    gpu_ids = [0]

    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'
    print(device)

    # record experiment parameters
    with open("experiment_params.txt", "a") as file_object:
        file_object.write("Exp #{}: {} epochs, batch {}, {} channels, {} levels, {} steps, {} samples.\n".format(args.expr_id, 
                                                                                                                 args.epochs, 
                                                                                                                 args.batch_size, 
                                                                                                                 args.num_channels, 
                                                                                                                 args.num_steps, 
                                                                                                                 args.num_samples))

    # getting data for training; just CIFAR10
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
            transforms.ToTensor()
    ])

    trainset = torchvision.datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(root='data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    if args.resume_training:
        if args.ckpt_path is "NONE":
            print("No path for the checkpoint file has been specified.")
        print("resuming training from checkpoint file: {}".format(args.ckpt_path))

    # define the model
    model = GlowModel(args.num_channels, args.num_levels, args.num_steps)
    model = model.to(device)
    model.describe()

    # if using GPU
    if device == 'cuda':
        model = torch.nn.DataParallel(model, gpu_ids)

    loss_function = NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.sched_warmup))

    times_array = []

    # training loop
    print("Starting training of the Glow model")
    for epoch in range(1, args.epochs + 1):
        print("Epoch [{} / {}]".format(epoch, args.epochs))
        start_time = time.time()
        train(epoch, model, trainloader, device, optimizer, scheduler, loss_function, args.grad_norm)
        test(epoch, model, testloader, device, loss_function, args.num_samples)
        elapsed_time = time.time() - start_time

        times_array.append(["Epoch " + str(epoch) + ": ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])

    with open("epoch_times.txt", "w") as txt_file:
        for line in times_array:
            txt_file.write(" ".join(line) + "\n")

best_loss = 0
global_loss = 0
main_wrapper()