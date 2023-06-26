'''Train DYNN from checkpoint of trained backbone'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


from models.custom_modules.confidence_score_threshold_gate import ConfidenceScoreThresholdGate
import os
import argparse
import mlflow

from collect_metric_iter import collect_metrics, get_empty_storage_metrics
from learning_helper import get_loss, get_dumb_loss, get_surrogate_loss, freeze_backbone as freeze_backbone_helper

from log_helper import log_metrics_mlflow
import numpy as np
from models import *
from timm.models import *
from utils import progress_bar
from timm.models import create_model
from utils import load_for_transfer_learning
from models.t2t_vit import TrainingPhase
from data_loading.data_loader_helper import get_cifar_10_dataloaders, CIFAR_10_IMG_SIZE, get_abs_path

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

parser.add_argument('--use_mlflow', default=True, help='Store the run with mlflow')
args = parser.parse_args()

freeze_backbone = True
transformer_layer_gating = [0, 1, 2, 3, 4, 5]
barely_train = False
G = len(transformer_layer_gating)

if barely_train:
    print('++++++++++++++WARNING++++++++++++++ you are barely training to test some things')

cfg = vars(args)
use_mlflow = args.use_mlflow
if use_mlflow:
    name = "_".join([str(a) for a in [args.dataset, args.batch]])
    print(name)
    project = 'DyNN'
    mlruns_path = get_abs_path(["mlruns"])
    mlflow.set_tracking_uri(mlruns_path)
    experiment = mlflow.set_experiment(project)
    mlflow.start_run(run_name=name)
    mlflow.log_params(cfg)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = 'cuda'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

train_loader, test_loader = get_cifar_10_dataloaders(train_batch_size=args.batch)
NUM_CLASSES = 10
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model(
    't2t_vit_7',
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

net.set_intermediate_heads(transformer_layer_gating)
net.set_learnable_gates(transformer_layer_gating)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


print('==> Resuming from checkpoint..')
assert os.path.isdir('../checkpoint'), 'Error: no checkpoint directory found!'
checkpoint = torch.load(args.ckp_path, map_location=torch.device(device))
param_with_issues = net.load_state_dict(checkpoint['net'], strict=False)
print("Missing keys:", param_with_issues.missing_keys)
print("Unexpected keys:", param_with_issues.unexpected_keys)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Backbone is always frozen
freeze_backbone_helper(net, ['intermediate_heads', 'gates'])
parameters = net.parameters()
def switch_training_phase(current_phase):
    if current_phase == TrainingPhase.GATE:
        return TrainingPhase.CLASSIFIER
    elif current_phase == TrainingPhase.CLASSIFIER:
        return TrainingPhase.GATE

optimizer = optim.SGD(parameters, lr=0.01,
                      momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=args.min_lr, T_max=60)

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
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Toggles requires grad to alternate training of classifiers and gates
        if bilevel_opt and batch_idx % bilevel_batch_count == 0:
            training_phase = switch_training_phase(training_phase)

        loss, things_of_interest = get_surrogate_loss(inputs, targets, optimizer, net, training_phase=training_phase)


        total += targets.size(0)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        stored_per_x, stored_metrics = collect_metrics(things_of_interest, G, targets, device,
                                                       stored_per_x, stored_metrics, training_phase)
        if training_phase == TrainingPhase.CLASSIFIER:
            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total_classifier+=targets.size(0)
            # compute metrics to display
            acc = 100. * correct / total_classifier

            loss = epoch_loss / (batch_idx + 1)
            progress_bar(
                batch_idx, len(train_loader),
                'Loss: %.3f | Classifier Acc: %.3f%% (%d/%d)' %
                (loss, acc, correct, total_classifier))

            if use_mlflow:
                log_dict = log_metrics_mlflow('train', acc, loss, len(transformer_layer_gating), stored_per_x,stored_metrics, total, total_classifier)


                mlflow.log_metrics(log_dict,
                                   step=batch_idx + (epoch * len(train_loader)))
        elif training_phase ==  TrainingPhase.GATE:
            total_gate+=targets.size(0)
            progress_bar(batch_idx, len(train_loader), 'Loss: %.3f ' % (loss))
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
        for batch_idx, (inputs, targets) in enumerate(test_loader):
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
                batch_idx, len(test_loader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (loss, acc, correct, total))

            if barely_train:
                if batch_idx>50:
                    print('++++++++++++++WARNING++++++++++++++ you are barely testing to test some things')
                    break
        if use_mlflow:
            log_dict = log_metrics_mlflow('test', acc, loss, len(transformer_layer_gating), stored_per_x,stored_metrics, total, total_classifier=total)
            mlflow.log_metrics(log_dict,
                               step=batch_idx + (epoch * len(train_loader)))
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