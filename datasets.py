import os
import torch
import torch.utils.data as data
import torchvision
from torchvision import transforms

def get_dataset(dataset_name, if_download, batch_size, num_workers):

    # defining transform that is applied to the training dataset part
    transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
    ])

    # defining the transform that is applied to hte testing dataset part
    transform_test = transforms.Compose([
            transforms.ToTensor()
    ])

    # specify the directory name for saving
    dir_name = os.path.join('data', dataset_name)

    print("Getting the dataset {}\n".format(dataset_name))
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=dir_name, train=True, download=if_download, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.CIFAR10(root=dir_name, train=False, download=if_download, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    if dataset_name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=dir_name, train=True, download=if_download, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root=dir_name, train=False, download=if_download, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if dataset_name == 'else':
        trainset = torchvision.datasets.MNIST(root=dir_name, train=True, download=if_download, transform=transform_train)
        trainloader = data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        testset = torchvision.datasets.MNIST(root=dir_name, train=False, download=if_download, transform=transform_test)
        testloader = data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return trainset, trainloader, testset, testloader