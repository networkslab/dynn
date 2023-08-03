import torch
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from matplotlib import pyplot as plt

CIFAR_10_IMG_SIZE = 32 * 32
CIFAR_100_IMG_SIZE = 32 * 32

RANDOM_SEED = 42
generator = torch.Generator().manual_seed(RANDOM_SEED)

class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., var=1.):
        self.var = var
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * (self.var**0.5) + self.mean

class FlattenTransform(torch.nn.Module):
    def __init__(self, start_dim = 0):
        self.start_dim = start_dim

    def __call__(self, tensor):
        return torch.flatten(tensor, start_dim=self.start_dim)
    
def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("dynn")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def get_cifar_10_dataloaders(img_size = 224, train_batch_size = 64, test_batch_size = 100, val_size = 0, noise_var=0.0, flatten=False):
    train_transforms_list = [
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size//8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        GaussianNoise(mean=0, var=noise_var),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24348, 0.26158)),
    ]

    test_transforms_list = [
        transforms.Resize(img_size),
        GaussianNoise(mean=0, var=noise_var),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24348, 0.26158)), # TODO compute these values for the test set
    ]
    if flatten:
        train_transforms_list.append(FlattenTransform(start_dim=0))
        test_transforms_list.append(FlattenTransform(start_dim=0))
    transform_train = transforms.Compose(train_transforms_list)

    transform_test = transforms.Compose(test_transforms_list)

    data_directory = get_abs_path(['data'])
    train_set = torchvision.datasets.CIFAR10(
        root=data_directory, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root=data_directory, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False)
    if val_size > 0:
        train_indices, val_indices = torch.utils.data.random_split(train_set, [len(train_set) - val_size, val_size], generator=generator)
        train_sampler = SubsetRandomSampler(train_indices.indices)
        valid_sampler = SubsetRandomSampler(val_indices.indices)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=valid_sampler)
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
        return train_loader, test_loader

def get_mnist_dataloaders(train_batch_size = 64, test_batch_size = 100, val_size = 0, noise_var=0.0, flatten=True):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    train_transforms_list = [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (MNIST_MEAN,), (MNIST_STD,)),
            GaussianNoise(noise_var)
    ]

    test_transforms_list = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            (MNIST_MEAN,), (MNIST_STD,)),
        GaussianNoise(noise_var)
    ]
    if flatten:
        train_transforms_list.append(FlattenTransform(start_dim=0))
        test_transforms_list.append(FlattenTransform(start_dim=0))
    transform_train = transforms.Compose(train_transforms_list)
    transform_test = transforms.Compose(test_transforms_list)

    data_directory = get_abs_path(['data'])

    train_set = torchvision.datasets.MNIST(root=data_directory, train=True, download=True, transform=transform_train)

    test_set = torchvision.datasets.MNIST(root=data_directory, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False)
    if val_size > 0:
        train_indices, val_indices = torch.utils.data.random_split(train_set, [len(train_set) - val_size, val_size], generator=generator)
        train_sampler = SubsetRandomSampler(train_indices.indices)
        valid_sampler = SubsetRandomSampler(val_indices.indices)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=train_sampler, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, sampler=valid_sampler)
        return train_loader, val_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
        return train_loader, test_loader

