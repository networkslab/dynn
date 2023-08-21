import torch
import os
from torch.utils.data._utils.collate import default_collate
import torchvision
import numpy as np

def get_path_to_project_root():
    cwd = os.getcwd()
    root_abs_path_index = cwd.split("/").index("dynn")
    return "/".join(os.getcwd().split("/")[:root_abs_path_index + 1])

def collate_remove_none(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))
    return default_collate(batch) if len(batch) > 0 else []

def get_abs_path(paths_strings):
    subpath = "/".join(paths_strings)
    src_abs_path = get_path_to_project_root()
    return f'{src_abs_path}/{subpath}/'

class FilterTransform(torch.nn.Module):
    def __init__(self, lower_threshold, upper_threshold):
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold

    def __call__(self, tensor):
        black_count = torch.sum(tensor > 0, dim=(1, 2)).item()
        if black_count >= self.lower_threshold and black_count <= self.upper_threshold:
            return tensor
        else:
            return None

def get_data_loaders(train_batch, test_batch):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=train_batch, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('.', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (MNIST_MEAN,), (MNIST_STD,))
                                   ])),
        batch_size=test_batch, shuffle=True)

    return train_loader, test_loader

def get_black_pixel_percentiles(dataset, transforms, samples_count = 20000, percentages = [0, 20, 40, 60, 80, 100]): # build statistics from 1000 images
    samples = dataset.train_data[:samples_count]
    dataset.transforms = torchvision.transforms.Compose(transforms)
    loader = torch.utils.data.DataLoader(dataset, samples_count)
    for X, y in loader:
        black_count = torch.sum((X.view(20000, 28, 28) > 0), dim=(1, 2))
        non_zero_count = black_count.numpy()
        percentiles = np.percentile(non_zero_count, percentages)
        break
    return percentiles

def get_thresholded_data_loaders(train_batch, test_batch, lower_perc, upper_perc):
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    data_directory = get_abs_path(['data'])
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(
                      (MNIST_MEAN,), (MNIST_STD,)),
                  FilterTransform(lower_perc, upper_perc)]
    train_dataset = torchvision.datasets.MNIST(data_directory, train=True, download=True,
                                               transform=torchvision.transforms.Compose(
                                                   transforms
                                               ))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch, shuffle=True, collate_fn=collate_remove_none)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(data_directory, train=False, download=True,
                                   transform=torchvision.transforms.Compose(
                                       transforms)),
        batch_size=test_batch, shuffle=True, collate_fn=collate_remove_none)

    return train_loader, test_loader