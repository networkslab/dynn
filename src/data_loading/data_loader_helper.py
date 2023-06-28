import torch
import torchvision
import torchvision.transforms as transforms
import os

CIFAR_10_IMG_SIZE = 32 * 32
CIFAR_100_IMG_SIZE = 32 * 32


def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("dynn")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

def get_cifar_10_dataloaders(img_size = 224, train_batch_size = 64, test_batch_size = 100):
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
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader

def get_cifar_100_dataloaders(img_size = 224, train_batch_size = 64, test_batch_size = 100):
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
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=train_batch_size, shuffle=True) # pass num_workers=n if multiprocessing is needed.
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=test_batch_size, shuffle=False)
    return train_loader, test_loader
