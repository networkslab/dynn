import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
from incremental_dataloading import get_data_loaders, get_thresholded_data_loaders, get_black_pixel_percentiles, get_abs_path
import json


class LogisticRegression(torch.nn.Module):
    def __init__(self, in_dim = 784, out_dim = 10):
        super(LogisticRegression, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_dim, out_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.view(-1, self.in_dim)
        linearized = self.linear(input)
        return linearized

def train_single_epoch(epoch, network, optimizer, criterion, train_loader, device):
    network.train()
    log_period = 300
    for batch_idx, (X, y) in enumerate(train_loader):
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        out = network(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        if batch_idx % log_period == 0:
            print(f'Epoch {epoch}, batch idx {batch_idx}, loss {loss}')

def test_single_epoch(network, criterion, test_loader, device):
    network.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(test_loader):
            X = X.to(device)
            y = y.to(device)
            out = network(X)
            count += y.shape[0]
            loss = criterion(out, y)
            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).sum()
        # print(f"count {count}, len {len(test_loader.dataset)}")
        acc = correct / count * 100
        return acc.item()

# network = LogisticRegression(784, 10)
# data_directory = get_abs_path(['data'])
# MNIST_MEAN = 0.1307
# MNIST_STD = 0.3081
# transforms = [torchvision.transforms.ToTensor(),
#               torchvision.transforms.Normalize(
#                   (MNIST_MEAN,), (MNIST_STD,))]
# train_dataset = torchvision.datasets.MNIST(data_directory, train=True, download=True,
#                                            transform=torchvision.transforms.Compose(
#                                                transforms
#                                            ))
# percentiles = get_black_pixel_percentiles(train_dataset, transforms, percentages=[0, 20])
# train_loader, test_loader = get_thresholded_data_loaders(64, 64, percentiles[0], percentiles[1])
# optimizer = optim.SGD(network.parameters(), lr=0.001,
#                       momentum=0.9)
# criterion = torch.nn.CrossEntropyLoss()
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# network = network.to(device)
#
# test_single_epoch(-1, network, criterion, test_loader, device)
# for e in range(10):
#     train_single_epoch(e, network, optimizer, criterion, train_loader, device)
#     test_single_epoch(e, network, criterion, test_loader, device)

def shifting_tasks(percentage_increments, percentage_range = 20):
    NUM_EPOCH = 3
    number_of_tasks = int((100 - percentage_range) / percentage_increments)
    result_dict = dict.fromkeys([f'task_{str(i)}' for i in range(number_of_tasks)])
    continual_network = LogisticRegression(784, 10)
    data_directory = get_abs_path(['data'])
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize(
                      (MNIST_MEAN,), (MNIST_STD,))]
    train_dataset = torchvision.datasets.MNIST(data_directory, train=True, download=True,
                                               transform=torchvision.transforms.Compose(
                                                   transforms
                                               ))
    percentages = [0, percentage_range]
    continual_optimizer = optim.SGD(continual_network.parameters(), lr=0.001,
                          momentum=0.9)

    criterion = torch.nn.CrossEntropyLoss()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    continual_network = continual_network.to(device)
    print(f"STARTING TRAINING WITH {number_of_tasks}")
    for t in range(number_of_tasks):
        print(f"SHIFTING TASK: Task number {t}, percentages are {percentages}")
        task_key = f'task_{str(t)}'
        reset_network = LogisticRegression(784, 10)
        reset_network = reset_network.to(device)
        reset_optimizer = optim.SGD(reset_network.parameters(), lr=0.001,
                            momentum=0.9)
        percentiles = get_black_pixel_percentiles(train_dataset, transforms, percentages=percentages)
        train_loader, test_loader = get_thresholded_data_loaders(64, 64, percentiles[0], percentiles[1])
        initial_acc_continual = test_single_epoch(continual_network, criterion, test_loader, device)
        initial_acc_reset = test_single_epoch(reset_network, criterion, test_loader, device)
        print(f"Initial accs: continual {initial_acc_continual}, reset {initial_acc_reset}")
        continual_accs = []
        reset_accs = []
        for e in range(NUM_EPOCH):
            train_single_epoch(e, continual_network, continual_optimizer, criterion, train_loader, device)
            continual_acc = test_single_epoch(continual_network, criterion, test_loader, device)
            train_single_epoch(e, reset_network, reset_optimizer, criterion, train_loader, device)
            reset_acc = test_single_epoch(reset_network, criterion, test_loader, device)
            continual_accs.append(continual_acc)
            reset_accs.append(reset_acc)
            print(f"Epoch {e}, continual acc {continual_acc}, reset acc {reset_acc}")
        result_dict[task_key] = {'continual_accs': continual_accs, 'reset_accs': reset_accs, 'percentages': percentages}
        write_result_dict(result_dict, 'accs_3.json')
        percentages = np.array(percentages) + percentage_increments
        percentages = percentages.tolist()

def write_result_dict(dict, file_name):
    results_path = get_abs_path(['src', 'plasticity_analysis', 'results'])
    file_name = f'{results_path}{file_name}'
    with open(file_name, 'w') as convert_file:
        convert_file.write(json.dumps(dict))

shifting_tasks(5, 30)
