# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

'''Tranfer pretrained T2T-ViT to downstream dataset: CIFAR10/CIFAR100.'''
import argparse
import os
import torch

import mlflow
import torch.backends.cudnn as cudnn
import torch.optim as optim
from timm.models import *
from timm.models import create_model

from data_loading.data_loader_helper import get_cifar_10_dataloaders
from utils import load_for_transfer_learning
from utils import progress_bar
from log_helper import setup_mlflow

parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or cifar100')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--transfer-ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--ckp-path', type=str, default='../checkpoint/checkpoint_cifar10_t2t_vit_7/ckpt_0.05_0.0005_90.47.pth',
                    help='path to checkpoint transfer learning model')
# us
parser.add_argument('--use_mlflow', default=True, help='Store the run with mlflow')
args = parser.parse_args()


use_mlflow = args.use_mlflow
if use_mlflow:
    name = "_".join([str(a) for a in [args.dataset, args.batch]])
    cfg = vars(args)
    setup_mlflow(name, cfg)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_loader, test_loader = get_cifar_10_dataloaders(train_batch_size=args.batch)
NUM_CLASSES = 10
MODEL = 't2t_vit_7'

print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model(
    MODEL,
    pretrained=False,
    num_classes=NUM_CLASSES,
    drop_rate=0.0,
    drop_connect_rate=None,
    drop_path_rate=0.1,
    drop_block_rate=None,
    global_pool=None,
    bn_tf=False,
    bn_momentum=None,
    bn_eps=None,
    img_size=224)


net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print('transfer learning, load t2t-vit pretrained model')
pretrained_model_weights = "../model_weights/71.7_T2T_ViT_7.pth.tar"
load_for_transfer_learning(net.module, pretrained_model_weights, use_ema=True, strict=False, num_classes=10)

print('set different lr for the t2t module, backbone and classifier(head) of T2T-ViT')
parameters = [{'params': net.module.tokens_to_token.parameters(), 'lr': args.transfer_ratio * args.lr},
              {'params': net.module.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
             {'params': net.module.head.parameters()}]


optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)

criterion = nn.CrossEntropyLoss()
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, _ = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        if use_mlflow:
            log_dict = {'train/loss': train_loss/(batch_idx+1), 'train/acc': 100.*correct/total}
            mlflow.log_metrics(log_dict, step=batch_idx)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{MODEL}'):
            os.mkdir(f'checkpoint_{args.dataset}_{MODEL}')
        torch.save(state, f'../checkpoint_{args.dataset}_{MODEL}/ckpt_{args.lr}_{args.wd}_{acc}.pth')
        best_acc = acc
    if use_mlflow:
        log_dict= {'best/test_acc': acc}
        mlflow.log_metrics(log_dict)
        mlflow.end_run()

for epoch in range(start_epoch, start_epoch+60):
    train(epoch)
    test(epoch)
    scheduler.step()
