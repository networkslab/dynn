import torch
import torchvision
import torchvision.transforms as transforms
import os
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
CIFAR_10_IMG_SIZE = 32 * 32
CIFAR_100_IMG_SIZE = 32 * 32

RANDOM_SEED = 42
generator = torch.Generator().manual_seed(RANDOM_SEED)
def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("dynn")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def get_cifar_10_dataloaders(img_size = 224, train_batch_size = 64, test_batch_size = 100, val_size = 0):
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size//8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24348, 0.26158)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.24703, 0.24348, 0.26158)), # TODO compute these values for the test set
    ])

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

def split_dataloader_in_n(data_loader, n):
    try:
        indices = data_loader.sampler.indices
    except:
        indices = list(range(len(data_loader.sampler)))
    dataset = data_loader.dataset
    list_indices = np.array_split(np.array(indices),n) 
    batch_size = data_loader.batch_size
    n_loaders = []
    for i in range(n):
        sampler = SubsetRandomSampler(list_indices[i])
        sub_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler)
        n_loaders.append(sub_loader)
    return n_loaders
    
def get_cifar_100_dataloaders(img_size = 32, train_batch_size = 64, test_batch_size = 100, val_size = 0):
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomCrop(img_size, padding=(img_size//8)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), # TODO compute these values for the test set
    ])

    data_directory = get_abs_path(['data'])
    train_set = torchvision.datasets.CIFAR100(
        root=data_directory, train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(
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
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
        return train_loader, test_loader

def get_svhn_dataloaders(train_batch_size = 64, test_batch_size = 100, val_size = 0):
    data_directory = get_abs_path(['data'])
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()]
    )
    train_set = torchvision.datasets.SVHN(
        root=data_directory, split="train", download=True, transform=transforms)
    test_set = torchvision.datasets.SVHN(
        root=data_directory, split="test", download=True, transform=transforms)
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
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
        return train_loader, test_loader
def get_latest_checkpoint_path(checkpoint_subpath):
    checkpoint_path = get_abs_path(["checkpoint", checkpoint_subpath])
    files = os.listdir(checkpoint_path)
    sorted_files = sorted(files)
    latest = sorted_files[-1]
    return f"{checkpoint_path}{latest}"
