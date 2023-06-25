# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.

'''Tranfer pretrained T2T-ViT to downstream dataset: CIFAR10/CIFAR100.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os
import argparse
import mlflow

from collect_metric_iter import collect_metrics, get_empty_storage_metrics
from learning_helper import  get_surrogate_loss
from log_helper import log_metrics_mlflow

from models import *
from timm.models import *
from utils import progress_bar
from timm.models import create_model
from utils import load_for_transfer_learning
from models.t2t_vit import TrainingPhase


parser = argparse.ArgumentParser(description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr', default=2e-4, type=float, help='minimal learning rate')
parser.add_argument('--dataset', type=str, default='cifar10',
                    help='cifar10 or cifar100')
parser.add_argument('--b', type=int, default=64,
                    help='batch size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--pretrained', action='store_true', default=False,
                    help='Start with pretrained version of specified network (if avail)')
parser.add_argument('--num-classes', type=int, default=10, metavar='N',
                    help='number of label classes (default: 1000)')
parser.add_argument('--model', default='t2t_vit_7', type=str, metavar='MODEL',
                    help='Name of model to train (default: "countception"')
parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                    help='Dropout rate (default: 0.0)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                    help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=0.1, metavar='PCT',
                    help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                    help='Drop block rate (default: None)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--img-size', type=int, default=224, metavar='N',
                    help='Image patch size (default: None => model default)')
parser.add_argument('--bn-tf', action='store_true', default=False,
                    help='Use Tensorflow BatchNorm defaults for models that support it (default: False)')
parser.add_argument('--bn-momentum', type=float, default=None,
                    help='BatchNorm momentum override (if not None)')
parser.add_argument('--bn-eps', type=float, default=None,
                    help='BatchNorm epsilon override (if not None)')
parser.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
# Transfer learning
parser.add_argument('--transfer-learning', default=False,
                    help='Enable transfer learning')
parser.add_argument('--transfer-model', type=str, default=None,
                    help='Path to pretrained model for transfer learning')
parser.add_argument('--transfer-ratio', type=float, default=0.01,
                    help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--ckp-path', type=str, default='checkpoint/checkpoint_cifar10_t2t_vit_7/ckpt_0.05_0.0005_90.47.pth',
                    help='path to checkpoint transfer learning model')
# us
parser.add_argument('--use_mlflow', default=True, help='Store the run with mlflow')
args = parser.parse_args()

freeze_backbone = True
transformer_layer_gating = [0,1,2,3,4,5]
barely_train = False
G = len(transformer_layer_gating)

if barely_train:
    print('++++++++++++++WARNING++++++++++++++ you are barely training to test some things')

cfg = vars(args)
use_mlflow = args.use_mlflow
if use_mlflow:
    name = "_".join([str(a) for a in [args.dataset, args.b]])
    print(name)
    project = 'DyNN'
    experiment = mlflow.set_experiment(project)
    mlflow.start_run(run_name=name)
    mlflow.log_params(cfg)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.RandomCrop(args.img_size, padding=(args.img_size//8)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(args.img_size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset=='cifar10':
    args.num_classes = 10
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

elif args.dataset=='cifar100':
    args.num_classes = 100
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
else:
    print('Please use cifar10 or cifar100 dataset.')

trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.b, shuffle=True) # pass num_workers=n if multiprocessing is needed.
testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False)

print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model(
    args.model,
    pretrained=args.pretrained,
    num_classes=args.num_classes,
    drop_rate=args.drop,
    drop_connect_rate=args.drop_connect, 
    drop_path_rate=args.drop_path,
    drop_block_rate=args.drop_block,
    global_pool=args.gp,
    bn_tf=args.bn_tf,
    bn_momentum=args.bn_momentum,
    bn_eps=args.bn_eps,
    checkpoint_path=args.initial_checkpoint,
    img_size=args.img_size)


if args.transfer_learning:
    print('transfer learning, load t2t-vit pretrained model')
    load_for_transfer_learning(net, args.transfer_model, use_ema=True, strict=False, num_classes=args.num_classes)

net.set_intermediate_heads(transformer_layer_gating)
net.set_learnable_gates(transformer_layer_gating)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.ckp_path, map_location=torch.device(device))
    param_with_issues = net.load_state_dict(checkpoint['net'], strict=False)
    print("Missing keys:", param_with_issues.missing_keys)
    print("Unexpected_keys keys:", param_with_issues.unexpected_keys)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



# set optimizer
if freeze_backbone:

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    from_num_params = sum([np.prod(p.size()) for p in model_parameters])
    # set everything to not trainable.
    for param in net.module.parameters():
        param.requires_grad = False
    # set the intermediate_heads params to trainable.
    for param in net.module.intermediate_heads.parameters():
        param.requires_grad = True

    for param in net.module.gates.parameters():
        param.requires_grad = True
    parameters = net.parameters()

    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    to_num_params = sum([np.prod(p.size()) for p in model_parameters])
    print('freeze the t2t module: from {} to {} trainable params.'.format(from_num_params,to_num_params ))

elif args.transfer_learning:
    print('set different lr for the t2t module, backbone and classifier(head) of T2T-ViT')
    parameters = [{'params': net.module.tokens_to_token.parameters(), 'lr': args.transfer_ratio * args.lr},
                  {'params': net.module.blocks.parameters(), 'lr': args.transfer_ratio * args.lr},
                 {'params': net.module.head.parameters()}]
else:
    parameters = net.parameters()

optimizer = optim.SGD(parameters, lr=args.lr,
                      momentum=0.9, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)


def switch_training_phase(current_phase):
    if current_phase == TrainingPhase.GATE:
        return TrainingPhase.CLASSIFIER 
    elif current_phase == TrainingPhase.CLASSIFIER:
        return TrainingPhase.GATE
    
def train(epoch, bilevel_opt = False, bilevel_batch_count = 20):
    print('\nEpoch: %d' % epoch)
    net.train()
    epoch_loss = 0
    correct = 0
    total = 0
    total_classifier = 0
    total_gate = 0
    training_phase =  TrainingPhase.GATE
    stored_per_x, stored_metrics = get_empty_storage_metrics(
        len(transformer_layer_gating))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Toggles requires grad to alternate training of classifiers and gates
        if bilevel_opt and batch_idx % bilevel_batch_count == 0:
            training_phase = switch_training_phase(training_phase)
            
        #     for param in net.module.intermediate_heads.parameters():
        #         param.requires_grad = not param.requires_grad
        #     print(f"Setting intermediate heads training to {list(net.module.intermediate_heads.parameters())[0].requires_grad}")
        #     for param in net.module.gates.parameters():
        #         param.requires_grad = not param.requires_grad
        #     print(f"Setting gates training to {list(net.module.gates.parameters())[0].requires_grad}")

        loss, things_of_interest  = get_surrogate_loss(inputs, targets, optimizer, net, training_phase=training_phase)
        

        total += targets.size(0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        stored_per_x, stored_metrics = collect_metrics(things_of_interest, G, targets, device,
            stored_per_x, stored_metrics, training_phase)
        if training_phase ==  TrainingPhase.CLASSIFIER: 
            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total_classifier+=targets.size(0)
            # compute metrics to display
            acc = 100. * correct / total_classifier
            
            loss = epoch_loss / (batch_idx + 1)
            progress_bar(
                batch_idx, len(trainloader),
                'Loss: %.3f | Classifier Acc: %.3f%% (%d/%d)' %
                (loss, acc, correct, total_classifier))

            if use_mlflow:
                log_dict = log_metrics_mlflow('train', acc, loss, len(transformer_layer_gating), stored_per_x,stored_metrics, total, total_classifier)
                

                mlflow.log_metrics(log_dict,
                                step=batch_idx + (epoch * len(trainloader)))
        elif training_phase ==  TrainingPhase.GATE: 
            total_gate+=targets.size(0)
            progress_bar(batch_idx, len(trainloader),'Loss: %.3f ' %(loss))
        if barely_train:
            if batch_idx>50:
                print('++++++++++++++WARNING++++++++++++++ you are barely training to test some things')
                break
    stored_metrics['acc'] = acc

    
    return stored_metrics


def test(epoch):

    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    stored_per_x, stored_metrics = get_empty_storage_metrics(
        len(transformer_layer_gating))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss, things_of_interest  = get_surrogate_loss(inputs, targets, optimizer, net)
            
           
            
            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total+=targets.size(0)
            
            test_loss += loss.item()
            stored_per_x, stored_metrics = collect_metrics(things_of_interest, G, targets, device,
            stored_per_x, stored_metrics, TrainingPhase.CLASSIFIER)
            

            acc = 100. * correct / total
            loss = test_loss / (batch_idx + 1)
            progress_bar(
                batch_idx, len(testloader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (loss, acc, correct, total))

            if barely_train:
                if batch_idx>50:
                    print('++++++++++++++WARNING++++++++++++++ you are barely testing to test some things')
                    break
        if use_mlflow:
            log_dict = log_metrics_mlflow('test', acc, loss, len(transformer_layer_gating), stored_per_x,stored_metrics, total, total_classifier=total)
            mlflow.log_metrics(log_dict,
                               step=batch_idx + (epoch * len(trainloader)))
    # Save checkpoint.
    
   
    
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{args.model}'):
            os.mkdir(f'checkpoint_{args.dataset}_{args.model}')
        torch.save(
            state,
            f'./checkpoint_{args.dataset}_{args.model}/ckpt_{args.lr}_{args.wd}_{acc}.pth'
        )
        best_acc = acc
    if use_mlflow:
        log_dict = {'best/test_acc': acc}
        mlflow.log_metrics(log_dict)
    return stored_metrics




for epoch in range(start_epoch, start_epoch + 5):
    stored_metrics_train = train(epoch, bilevel_opt=True, bilevel_batch_count=100)
    stored_metrics_test = test(epoch)
    scheduler.step()

mlflow.end_run()
