## Standard libraries
import os
import time
import numpy as np
import argparse
import sys

# pytorch
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as sched

# Torchvision
import torchvision

from glow_model import GlowModel
from utilities import *
from datasets import *
from flows import *

# enablig grad for loss calc
@torch.enable_grad()
def train(epoch, model, trainloader, device, optimizer, scheduler, loss_func, max_grad_norm):
    print("===> EPOCH {}".format(epoch))
    global global_step
    # initialising training mode; just so the model "knows" it is training
    model.train()
    loss_meter = AvgMeter()
    
    # training on batches
    for x, _ in trainloader:
        x = x.to(device)
        optimizer.zero_grad()
        # forward pass so reverse mode is turned off
        z, sum_lower_det_jacobian = model(x, reverse=False)
        # calculating and updating loss
        loss = loss_func(z, sum_lower_det_jacobian)
        loss_meter.update(loss.item(), x.size(0))
        loss.backward()

        # clip gradient if too much
        if max_grad_norm > 0:
            clip_grad_norm(optimizer, max_grad_norm)

        # advance optimizer and scheduler 
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
        z, sum_lower_det_jacobian = model(x, reverse=False)
        loss = loss_func(z, sum_lower_det_jacobian)
        loss_meter.update(loss.item(), x.size(0))

    # checking if a checpoint should be saves
    if epoch % args.ckpt_interval == 0:
        print('Saving checkpoint file from the epoch #{}'.format(epoch))

    # checking if this current epoch is a saving interval
    if epoch % args.img_interval == 0:
        print("saving images from the epoch #{}".format(epoch))
        images = sample(model, num_samples, device) 
        path_to_images = 'samples/epoch_' + str(epoch) # custom name for each epoch
        os.makedirs(path_to_images, exist_ok=True) # create a dir for each epoch

        # save every image separately with an unique name
        for i in range(images.size(0)):
            torchvision.utils.save_image(images[i, :, :, :], '{}/img_{}.png'.format(path_to_images, i))

        # check if this epoch a grid should be saved
        if epoch % args.grid_interval == 0:
            print('saving nice grid')
            # save a grid of images
            images_concat = torchvision.utils.make_grid(images, nrow=int(num_samples ** 0.5), padding=2, pad_value=255)
            torchvision.utils.save_image(images_concat, 'samples/grid_epoch_{}.png'.format(epoch))

if __name__ == '__main__':
    # parsing args for easier running of the program
    parser = argparse.ArgumentParser()
    
    # model parameters
    parser.add_argument('--model', type=str, default='glow', help='Name of the model in use.')
    parser.add_argument('--num_channels', type=int, default=512, help='Number of channels.')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of flow levels.')
    parser.add_argument('--num_steps', type=int, default=16, help='Number of flow steps.')
    # optimizer and scheduler parameters
    parser.add_argument('--lr', type=float, default=1e-9, help='Learning rate for the optimizer.')
    parser.add_argument('--grad_norm', type=float, default=-1, help="Maximum value of gradient.")
    parser.add_argument('--sched_warmup', type=int, default=500000, help='Warm-up period for scheduler.')
    # training parameters
    parser.add_argument('--gpu', action='store_true', default=False, help='Flag indicating GPU use.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--resume_training', action='store_true', default=False, help='Flag indicating resuming training from checkpoint.')
    parser.add_argument('--num_samples', type=int, default=64, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    # dataset 
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'], help='Choose dataset: [mnist/cifar10]')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for datasets.')
    parser.add_argument('--download', action='store_true', default=False, help='Flad indicating when a dataset should be downloaded.')
    # checkpointing and img saving
    parser.add_argument('--ckpt_interval', type=int, default=1, help='Create a checkpoint file every N epochs.')
    parser.add_argument('--img_interval', type=int, default=1, help='Generate images every N epochs.')
    parser.add_argument('--ckpt_path', type=str, default='NONE', help='Path to the checkpoint file to use.')
    parser.add_argument('--grid_interval', type=int, default=50, help='How often to save images in a nice grid.')

    args = parser.parse_args()

    # initialising variables for keeping track of the global step and the best loss so far
    best_loss = 0
    global_step = 0

    # training on GPU if possible
    device = 'cuda' if torch.cuda.is_available() and args.gpu else 'cpu'

    # get data for training according to the specified dataset name
    trainset, trainloader, testset, testloader = get_dataset(args.dataset, args.download, args.batch_size, args.num_workers)
    
    # define the model
    if args.model == 'glow':
        model = GlowModel(4*3, args.num_channels, args.num_levels, args.num_steps)
        model = model.to(device)

    # account for training continuation; if incorrect checkpoint file name terminate the program with an appropriate error message
    if args.resume_training:
        if args.ckpt_path == "NONE":
            # print("No path for the checkpoint file has been specified. Training will start without any checkpoint.")
            sys.exit("Chechpoint file must be specified when continuing training.")
        # check if file exists
        if not os.path.isfile(args.ckpt_path):
            # print("The checkpoint path is incorrect! Training will start without any checkpoint.")
            sys.exit("Path to the checkpoint file is incorrect, specify correct path to the file.")
        # use save cofiguration
        else:
            checkpoint_loaded = torch.load(args.ckpt_path)
            model.load_state_dict(checkpoint_loaded['state_dict'])
            # global best_loss
            # global global_step
            best_loss = checkpoint_loaded['test_loss']
            starting_epoch = checkpoint_loaded['epoch']
            global_step = starting_epoch * len(trainset)
        print("resuming training from checkpoint file: {}".format(args.ckpt_path))

    # defining the loss function, the optimizer, and the scheduler
    # kinda established things
    loss_function = NLLLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.sched_warmup))

    # for reference
    times_array = []

    # training loop repeating for a specified number of epochs; starts from #1 in order to start naming epochs from 1
    print("Starting training of the Glow model")
    for epoch in range(1, args.epochs + 1):
        print("Epoch [{} / {}]".format(epoch, args.epochs))

        # measuring epoch execution time for reference
        start_time = time.time()

        # each epoch consist of training part and testing part
        train(epoch, model, trainloader, device, optimizer, scheduler, loss_function, args.grad_norm)
        test(epoch, model, testloader, device, loss_function, args.num_samples, args)

        elapsed_time = time.time() - start_time
        # recording time per epoch to a dataframe
        times_array.append(["Epoch " + str(epoch) + ": ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
    
    print("the training is finished.")