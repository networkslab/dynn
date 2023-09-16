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

import torchvision
import torchvision.transforms as transforms
from timm.models import *
from timm.models import create_model
from data_loading.data_loader_helper import get_cifar_10_dataloaders, get_cifar_100_dataloaders, get_path_to_project_root, get_svhn_dataloaders, get_abs_path
from utils import load_for_transfer_learning
from utils import progress_bar
from log_helper import setup_mlflow
from models import *
from models.register_models import *
parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or cifar100 or svhn')
parser.add_argument('--batch', type=int, default=64,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--transfer-ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--use_mlflow', default=True, help='Store the run with mlflow')
args = parser.parse_args()


use_mlflow = args.use_mlflow
if use_mlflow:
    name = "_".join([str(a) for a in [args.dataset, args.batch]])
    cfg = vars(args)
    setup_mlflow(name, cfg, "svhn")
path_project = get_path_to_project_root()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


if args.dataset=='cifar10':
    NUM_CLASSES = 10
    img_size = 224
    train_loader, test_loader = get_cifar_10_dataloaders(img_size = img_size,train_batch_size=args.batch, test_batch_size=args.batch)
    
    #pretrained_model_weights = os.path.join(path_project,"model_weights/81.5_T2T_ViT_14.pth.tar")
    pretrained_model_weights = os.path.join(path_project,"model_weights/71.7_T2T_ViT_7.pth.tar")
    MODEL = 't2t_vit_7'
    if args.resume:
        checkpoint = torch.load(os.path.join(path_project, 'checkpoint/checkpoint_cifar10_t2t_vit_7/ckpt_0.01_0.0005_94.95.pth'))
elif args.dataset=='cifar100':
    NUM_CLASSES = 100
    img_size = 224
    train_loader, test_loader = get_cifar_100_dataloaders(img_size = img_size,train_batch_size=args.batch)
    
    pretrained_model_weights = os.path.join(path_project,"model_weights/81.5_T2T_ViT_14.pth.tar")
    if args.resume:
        checkpoint = torch.load(os.path.join(path_project, 'checkpoint/cirfar100_t2t-vit-14_88.4.pth'))
    MODEL = 't2t_vit_14'
elif args.dataset=='svhn':
    NUM_CLASSES = 10
    img_size = 32
    train_loader, test_loader = get_svhn_dataloaders(train_batch_size=args.batch)

    pretrained_model_weights = os.path.join(path_project,"model_weights/71.7_T2T_ViT_7.pth.tar")
    # if args.resume:
    #     checkpoint = torch.load(os.path.join(path_project, 'checkpoint/cirfar100_t2t-vit-14_88.4.pth'))
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
    img_size=img_size)


net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True
print('transfer learning, load t2t-vit pretrained model')

load_for_transfer_learning(net.module, pretrained_model_weights, use_ema=True, strict=False, num_classes=NUM_CLASSES)

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    param_with_issues = net.load_state_dict(checkpoint['net'], strict=False)
    print("Missing keys:", param_with_issues.missing_keys)
    print("Unexpected_keys keys:", param_with_issues.unexpected_keys)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



print('set different lr for the t2t module, backbone and classifier(head) of T2T-ViT')
parameters = [{'params': net.module.tokens_to_token.parameters(), 'lr': args.transfer_ratio * args.lr},
              {'params': net.module.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
             {'params': net.module.head.parameters()}]


optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)

criterion = torch.nn.CrossEntropyLoss()
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
        outputs, _, _ = net(inputs)
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
            outputs, _, _ = net(inputs)
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
        checkpoint_folder_path = get_abs_path(["checkpoint"])
        target_checkpoint_folder_path = f'{checkpoint_folder_path}/checkpoint_{args.dataset}_{MODEL}'
        if not os.path.isdir(target_checkpoint_folder_path):
            os.mkdir(target_checkpoint_folder_path)
        torch.save(state, f'{target_checkpoint_folder_path}/ckpt_{args.lr}_{args.wd}_{acc}.pth')
        best_acc = acc
    if use_mlflow:
        log_dict= {'best/test_acc': acc}
        mlflow.log_metrics(log_dict)
        mlflow.end_run()

for epoch in range(start_epoch, start_epoch+60):
    train(epoch)
    test(epoch)
    scheduler.step()
