## Standard libraries
import os
import time
import argparse
import sys
import math

# pytorch
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as sched

# Torchvision
import torchvision

from glow_model import GlowModel
from utilities import *
from datasets import *
# from flows import *
from tqdm import tqdm

# getting a sample of n (num_samples) images from latent space
@torch.no_grad()
def sample(model, device, args):
    # get a specified number of tensors in the shape of a desired images from the normal random distribution
    z = torch.randn((args.num_samples, args.num_features, args.img_height, args.img_width), dtype=torch.float32, device=device)
    # use the invertibility principle to get the sample
    imgs, _ = model(z, reverse=True)
    imgs = torch.sigmoid(imgs)
    return imgs

# enablig grad for loss calc
@torch.enable_grad()
def train(epoch, model, trainloader, device, optimizer, scheduler, loss_func, max_grad_norm, max_grad_clip):
    print("\t-> TRAIN")
    # initialising training mode; just so the model "knows" it is training
    global global_step
    local_step = 0
    model.train()
    # initialising counter for loss calculations
    loss_meter = AvgMeter()

    # fancy progress bar
    with tqdm(total=len(trainloader.dataset)) as progress_bar:
        for x, _ in trainloader:
            x = x.to(device)
            optimizer.zero_grad()
            # forward pass so reverse mode is turned off
            z, sldj = model(x, reverse=False)
            # calculating and updating loss
            current_loss = loss_func(z, sldj)
            loss_meter.update(current_loss.item(), x.size(0))
            # backprop loss
            model.zero_grad()
            current_loss.backward()
            # if local_step % 2048 == 0:
            #     plot_grad_flow(model.named_parameters(), local_step, epoch)

            # clip gradient if too much
            # if max_grad_norm > 0:
            #     clip_grad_norm(optimizer, max_grad_norm)
            
            # v1
            torch.nn.utils.clip_grad_value_(model.parameters(), 5)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
            
            # v2
            # if max_grad_clip > 0:
            #     torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
            # if max_grad_norm > 0:
            #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # if local_step % 2048 == 0:
            #     plot_grad_flow(model.named_parameters(), local_step, epoch, after=True)

            # advance optimizer and scheduler and update parameters
            optimizer.step()
            scheduler.step(global_step)

            progress_bar.set_postfix(nll=loss_meter.avg,
                                    bpd=bits_per_dimension(z, loss_meter.avg),
                                    lr=optimizer.param_groups[0]['lr'])
            progress_bar.update(x.size(0))

            # updating the global step using the batch size used for training
            global_step += x.size(0)
            local_step += x.size(0)
    # return global_step

@torch.no_grad()
def test(epoch, model, testloader, device, loss_func, args):
# def test(epoch, model, testloader, device, optimizer, scheduler, loss_func, best_loss, args):
    print("\t-> TEST")
    global best_loss
    # setting a flag for indicating if this epoch is best ever
    best = False
    model.eval()
    loss_meter = AvgMeter()

    for x, _ in testloader:
        x = x.to(device)
        z, sldj = model(x, reverse=False)
        current_loss = loss_func(z, sldj)
        loss_meter.update(current_loss.item(), x.size(0))

    if loss_meter.avg < best_loss:
        print('Updating best loss: [{}] -> [{}]'.format(best_loss, loss_meter.avg))
        best_loss = loss_meter.avg
        # indicating this epoch has achieved the best loss value so far
        best = True
    
    # save checkpoint file on interval
    if epoch % args.ckpt_interval == 0:
        print('Saving checkpoint file from the epoch #{}'.format(epoch))
        save_model_checkpoint(model, epoch, args.dataset, loss_meter.avg, best)

    # Save samples and data on the specified interval
    if epoch % args.img_interval == 0:
        print("Saving images from the epoch #{}".format(epoch))

        # getting a sample of n images
        images = sample(model, device, args)
        # creating a path to an epoch directory so the images are sorted by epoch
        path_to_images = 'samples/epoch_' + str(epoch)
        # deciding if saving images to grid
        save_grid = epoch % args.grid_interval == 0
        save_sampled_images(epoch, images, args.num_samples, path_to_images, if_grid=save_grid)

    # return best_loss

if __name__ == '__main__':

    # parsing args for easier running of the program
    parser = argparse.ArgumentParser()
    
    # model parameters
    parser.add_argument('--model', type=str, default='glow', help='Name of the model in use.')
    parser.add_argument('--hidden_layers', type=int, default=512, help='Number of channels.')
    parser.add_argument('--num_levels', type=int, default=3, help='Number of flow levels.')
    parser.add_argument('--num_steps', type=int, default=32, help='Number of flow steps.')
    # optimizer and scheduler parameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for the optimizer.')
    parser.add_argument('--max_grad_norm', type=float, default=-1., help="Maximum value of gradient.")
    parser.add_argument('--max_grad_clip', type=float, default=0, help="Maximum value of gradient.")
    parser.add_argument('--sched_warmup', type=int, default=500000, help='Warm-up period for scheduler.')
    # training parameters
    parser.add_argument('--no_gpu', action='store_true', default=False, help='Flag indicating no GPU use.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs.')
    parser.add_argument('--load_model', action='store_true', default=False, help='Flag indicating loading a model from specified checkpoint.')
    parser.add_argument('--num_samples', type=int, default=16, help='Number of samples.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training.')
    parser.add_argument('--usage_mode', type=str, default='train', help='What mode to run the program in [train/sample] When sampling a path to a checkpoint file MUST be specified.')
    # dataset 
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['mnist', 'cifar10', 'chest_xray'], help='Choose dataset: [mnist/cifar10/chest_xray]')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for datasets.')
    parser.add_argument('--download', action='store_true', default=False, help='Flag indicating when a dataset should be downloaded.')
    # checkpointing and img saving
    parser.add_argument('--ckpt_interval', type=int, default=1, help='Create a checkpoint file every N epochs.')
    parser.add_argument('--img_interval', type=int, default=1, help='Generate images every N epochs.')
    parser.add_argument('--ckpt_path', type=str, default='NONE', help='Path to the checkpoint file to use.')
    parser.add_argument('--grid_interval', type=int, default=1, help='How often to save images in a nice grid.')
    # image params 
    parser.add_argument('--num_features', type=int, default=3, help='Number of spatial channels of an image [cifar10: 3 / mnist: 1].')
    parser.add_argument('--img_height', type=int, default=32, help='Image height in pixels [cifar10: 32 / mnist: 28]')
    parser.add_argument('--img_width', type=int, default=32, help='Image width in pixels [cifar10: 32 / mnist: 28]')

    # python main.py --epochs 10 --download

    args = parser.parse_args()

    # training on GPU if possible
    device = 'cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu'

    # get data for training according to the specified dataset name
    trainset, trainloader, testset, testloader = get_dataset(args.dataset, args.download, args.batch_size, args.num_workers)

    if args.dataset == 'cifar10':
        args.num_features = 3
        args.img_height = 32
        args.img_width = 32
    elif args.dataset == 'mnist':
        args.num_features = 1
        args.img_height = 28
        args.img_width = 28
    elif args.dataset == 'chest_xray':
        args.num_features = 3
        args.img_height = 32
        args.img_width = 32
    else:
        sys.exit('Incorrect dataset name')

    # define the model
    if args.model == 'glow':
        model = GlowModel(args.num_features, args.hidden_layers, args.num_levels, args.num_steps, args.img_height, args.img_width)
        model = model.to(device)

    print('The model has been created. It looks like this:\n')
    model.describe()

    # optimizer takes care of updating the parameters of the model after each batch
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # scheduler takes care of the adjustment of the learning rate
    scheduler = sched.LambdaLR(optimizer, lambda s: min(1., s / args.sched_warmup))

    # account for training continuation; if incorrect checkpoint file name terminate the program with an appropriate error message
    if args.load_model:
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
            optimizer.load_state_dict(checkpoint_loaded['optim'])
            scheduler.load_state_dict(checkpoint_loaded['sched'])
            best_loss = checkpoint_loaded['test_loss']
            starting_epoch = checkpoint_loaded['epoch']
            global_step = starting_epoch * len(trainset)
            print('Loading checkpoint file with best loss: {}, starting epoch: {}, and global step: {}'.format(best_loss, starting_epoch, global_step))
        print("resuming training from checkpoint file: {}".format(args.ckpt_path))
    else:
        # if training from scratch then init default values
        # initialising variables for keeping track of the global step and the best loss so far
        global_step = 0
        best_loss = math.inf
        # best_loss = 0
        starting_epoch = 1

    # run in training mode
    if args.usage_mode == 'train':
        # defining the loss function, the optimizer, and the scheduler
        # kinda established things
        loss_function = NLLLoss().to(device)

        # for reference
        times_array = []

        # training loop repeating for a specified number of epochs; starts from #1 in order to start naming epochs from 1
        print("Starting training of the Glow model")
        for epoch in range(starting_epoch, args.epochs + starting_epoch):
            print("=============> EPOCH [{} / {}]".format(epoch, args.epochs))

            # measuring epoch execution time for reference
            start_time = time.time()

            # each epoch consist of training part and testing part
            # new_global_step = train(epoch, model, trainloader, device, optimizer, scheduler, loss_function, args.grad_norm, global_step)
            # new_best_loss = test(epoch, model, testloader, device, optimizer, scheduler, loss_function, best_loss, args)
            train(epoch, model, trainloader, device, optimizer, scheduler, loss_function, args.max_grad_norm, args.max_grad_clip)
            test(epoch, model, testloader, device, loss_function, args)
            # global_step, best_loss = new_global_step, new_best_loss

            elapsed_time = time.time() - start_time
            # recording time per epoch to a dataframe
            times_array.append(["Epoch " + str(epoch) + ": ", time.strftime("%H:%M:%S", time.gmtime(elapsed_time))])
        
        print("the training is finished.")
    # run in sampling mode
    elif args.usage_mode == 'sample':
        print('sampling')
        images = sample(model, device, args)
        path_to_images = 'samples/{}/epoch_{}'.format(args.dataset, starting_epoch) # custom name for each epoch
        save_sampled_images(starting_epoch, images, args.num_samples, path_to_images, if_grid=True)
    # incorrect mode
    else:
        model.describe()
        sys.exit('Incorrect usage mode! Try --usage_mode train/sample')