'''Train DYNN from checkpoint of trained backbone'''
import argparse
from cmath import nan
import os
import torch
import mlflow
import torch.backends.cudnn as cudnn
import torch.optim as optim
from timm.models import *
from timm.models import create_model
import numpy as np
from collect_metric_iter import collect_metrics, get_empty_storage_metrics
from data_loading.data_loader_helper import get_abs_path, get_cifar_10_dataloaders, get_path_to_project_root, get_cifar_100_dataloaders
from learning_helper import get_loss, get_surrogate_loss, get_boosted_loss, freeze_backbone as freeze_backbone_helper
from log_helper import log_metrics_mlflow, setup_mlflow, compute_gated_accuracy
from models.custom_modules.gate import GateType
from utils import progress_bar
from early_exit_utils import switch_training_phase
from models.t2t_vit import TrainingPhase, GateTrainingScheme, GateSelectionMode

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr',default=2e-4,type=float,help='minimal learning rate')
parser.add_argument('--dataset',type=str,default='cifar10',help='cifar10 or cifar100')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--ce_ic_tradeoff',default=0.001,type=float,help='cost inference and cross entropy loss tradeoff')
parser.add_argument('--G', default=6, type=int, help='number of gates')
parser.add_argument('--num_epoch', default=5, type=int, help='num of epochs')
parser.add_argument('--warmup_batch_count',default=50,type=int,help='number of batches for warmup where all classifier are trained')
parser.add_argument('--bilevel_batch_count',default=200,type=int,help='number of batches before switching the training modes')
parser.add_argument('--barely_train',action='store_true',help='not a real run')
parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--model', type=str,default='learn_gate_direct')  # learn_gate, learn_gate_direct
parser.add_argument('--gate',type=GateType,default=GateType.CODE,choices=GateType)  # unc, code, code_and_unc
parser.add_argument('--drop-path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: None)')
parser.add_argument('--gate_selection_mode', type=GateSelectionMode, default=GateSelectionMode.PROBABILISTIC, choices=GateSelectionMode)
parser.add_argument('--transfer-ratio',type=float,default=0.01, help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--ckp-path',type=str,default='checkpoint_cifar10_t2t_vit_7/ckpt_0.05_0.0005_90.47.pth',help='path to checkpoint transfer learning model')
parser.add_argument('--gate_training_scheme',default='DEFAULT',help='Gate training scheme (how to handle gates after first exit)',
    choices=['DEFAULT', 'IGNORE_SUBSEQUENT', 'EXIT_SUBSEQUENT'])
parser.add_argument('--proj_dim',default=32,help='Target dimension of random projection for ReLU codes')
parser.add_argument('--use_mlflow',default=True,help='Store the run with mlflow')
args = parser.parse_args()

transformer_layer_gating = [g for g in range(args.G)]

proj_dim = int(args.proj_dim)
if args.barely_train:
    print(
        '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
    )
gate_training_scheme = GateTrainingScheme[args.gate_training_scheme]
use_mlflow = args.use_mlflow
if use_mlflow:
    name = "_".join([
        str(a) for a in [
            args.model, args.ce_ic_tradeoff, args.gate,
            args.gate_training_scheme
        ]
    ])
    cfg = vars(args)
    setup_mlflow(name, cfg)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

checkpoint_path = os.path.join(get_path_to_project_root(), 'checkpoint')
assert os.path.isdir(checkpoint_path)
checkpoint_path_to_load = os.path.join(checkpoint_path, args.ckp_path)

IMG_SIZE = 224
train_loader, test_loader = get_cifar_10_dataloaders(
    img_size=IMG_SIZE, train_batch_size=args.batch, test_batch_size=args.batch)
NUM_CLASSES = 10
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model('t2t_vit_7_boosted',
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
                   img_size=IMG_SIZE)
net.set_CE_IC_tradeoff(args.ce_ic_tradeoff)
net.set_intermediate_heads(transformer_layer_gating)
net.set_gate_training_scheme_and_mode(gate_training_scheme, args.gate_selection_mode)

direct_exit_prob_param = args.model == 'learn_gate_direct'
net.set_learnable_gates(device,
                        transformer_layer_gating,
                        direct_exit_prob_param=direct_exit_prob_param,
                        gate_type=args.gate,
                        proj_dim=proj_dim)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = os.path.join(get_path_to_project_root(), 'checkpoint')
assert os.path.isdir(checkpoint_path)
checkpoint_path_to_load = os.path.join(checkpoint_path, args.ckp_path)
checkpoint = torch.load(checkpoint_path_to_load,
                        map_location=torch.device(device))
param_with_issues = net.load_state_dict(checkpoint['net'], strict=False)
print("Missing keys:", param_with_issues.missing_keys)
print("Unexpected keys:", param_with_issues.unexpected_keys)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Backbone is always frozen
freeze_backbone_helper(net, ['intermediate_heads', 'gates'])
parameters = net.parameters()
optimizer = optim.SGD(parameters,
                      lr=args.lr,
                      momentum=0.9,
                      weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                       eta_min=args.min_lr,
                                                       T_max=args.num_epoch)


def train(epoch,
          bilevel_opt=False,
          bilevel_batch_count=20,
          classifier_warmup_periods=0):
    print('\nEpoch: %d' % epoch)
    net.train()
    gate_loss = 0
    classifier_loss = 0
    correct = 0
    total = 0
    total_classifier = 0
    total_gate = 0
    training_phase = TrainingPhase.WARMUP
    stored_per_x, stored_metrics = get_empty_storage_metrics(args.G)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        if classifier_warmup_periods > 0 and batch_idx < classifier_warmup_periods:
            training_phase = TrainingPhase.WARMUP
        elif classifier_warmup_periods > 0 and batch_idx == classifier_warmup_periods:  # only hit when we switch from warmup to normal
            # clean slate, we set every counter to zero
            gate_loss = 0
            classifier_loss = 0
            correct = 0
            total = 0
            total_classifier = 0
            total_gate = 0
            stored_per_x, stored_metrics = get_empty_storage_metrics(args.G)
            training_phase = switch_training_phase(training_phase)
        elif bilevel_opt and batch_idx % bilevel_batch_count == 0:
            if net.module.are_all_classifiers_frozen(): # no need to train classifiers anymore
                training_phase = TrainingPhase.GATE
                print("All classifiers are frozen, setting training phase to gate")
            else:
                training_phase = switch_training_phase(training_phase)

        if training_phase == TrainingPhase.WARMUP:
            loss, things_of_interest = get_loss(inputs, targets, optimizer,
                                                net)
        else:
            loss, things_of_interest = get_surrogate_loss(
                inputs, targets, optimizer, net, training_phase=training_phase)

        total += targets.size(0)
        loss.backward()
        optimizer.step()
        
        stored_per_x, stored_metrics = collect_metrics(things_of_interest,
                                                       args.G, targets, device,
                                                       stored_per_x,
                                                       stored_metrics,
                                                       training_phase)
        if training_phase == TrainingPhase.CLASSIFIER:
            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total_classifier += targets.size(0)
            # compute metrics to display
            gated_acc = 100. * correct / total_classifier
            classifier_loss += loss.item()
            loss = classifier_loss / total_classifier
            progress_bar(
                batch_idx, len(train_loader),
                'Classifier Loss: %.3f | Classifier Acc: %.3f%% (%d/%d)' %
                (loss, gated_acc, correct, total_classifier))

            stored_metrics['acc'] = gated_acc
            if use_mlflow:
                log_dict = log_metrics_mlflow(
                    prefix_logger='train',
                    gated_acc=gated_acc,
                    loss=loss,
                    G=args.G,
                    stored_per_x=stored_per_x,
                    stored_metrics=stored_metrics,
                    total_classifier=total_classifier,
                    batch=targets.size(0))

                mlflow.log_metrics(log_dict,
                                   step=batch_idx +
                                   (epoch * len(train_loader)))
        elif training_phase == TrainingPhase.WARMUP:
            total_classifier += targets.size(0)
            classifier_loss += loss.item()
            loss = classifier_loss / total_classifier
            progress_bar(batch_idx, len(train_loader),
                         'Loss: %.3f | Warmup  time' % (loss))

            if use_mlflow:
                log_dict = log_metrics_mlflow(
                    prefix_logger='train',
                    gated_acc=None,
                    loss=loss,
                    G=args.G,
                    stored_per_x=stored_per_x,
                    stored_metrics=stored_metrics,
                    total_classifier=total_classifier,
                    batch=targets.size(0))

                mlflow.log_metrics(log_dict,
                                   step=batch_idx +
                                   (epoch * len(train_loader)))

        elif training_phase == TrainingPhase.GATE:
            total_gate += targets.size(0)
            gate_loss += loss.item()
            loss = gate_loss / total_gate
            progress_bar(batch_idx, len(train_loader), 'Gate Loss: %.3f ' % (loss))
            exit_count_optimal_gate_perc = {k: v / total_gate * 100 for k, v in stored_metrics['exit_count_optimal_gate'].items()}
            gate_exit_acc = stored_metrics['correct_exit_count'] / (total_gate * args.G) * 100
            log_dict ={'train/gate_loss':loss, 'train/gate_exit_acc': gate_exit_acc}
            for g in range(args.G):
                log_dict['train' + '/optimal_percent_exit' +
                         str(g)] = exit_count_optimal_gate_perc[g]
            mlflow.log_metrics(log_dict,
                               step=batch_idx + (epoch * len(train_loader)))
        if args.barely_train:
            if batch_idx > 50:
                print(
                    '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
                )
                break

    return stored_metrics

def train_boosted(epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        boosted_loss = get_boosted_loss(inputs, targets, optimizer, net.module)
        boosted_loss.backward()
        optimizer.step()
        progress_bar(
            batch_idx, len(train_loader),
            'Classifier Loss: %.3f' % boosted_loss)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    stored_per_x_classifier, stored_metrics_classifier = get_empty_storage_metrics(
        len(transformer_layer_gating))
    stored_per_x_gate, stored_metrics_gate = get_empty_storage_metrics(
        len(transformer_layer_gating))
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            loss, things_of_interest = get_surrogate_loss(
                inputs, targets, optimizer, net)

            gated_y_logits = things_of_interest['gated_y_logits']
            _, predicted = gated_y_logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

            test_loss += loss.item()
            stored_per_x_classifier, stored_metrics_classifier = collect_metrics(things_of_interest, args.G, targets, device,
                                                           stored_per_x_classifier, stored_metrics_classifier, TrainingPhase.CLASSIFIER)
            stored_per_x_gate, stored_metrics_gate = collect_metrics(things_of_interest, args.G, targets, device,
                                                                                 stored_per_x_gate, stored_metrics_gate, TrainingPhase.GATE)

            gated_acc = 100. * correct / total
            loss = test_loss / (batch_idx + 1)
            progress_bar(
                batch_idx, len(test_loader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (loss, gated_acc, correct, total))

            if args.barely_train:
                if batch_idx > 50:
                    print(
                        '++++++++++++++WARNING++++++++++++++ you are barely testing to test some things'
                    )
                    break
        # Decide whether to freeze classifiers or not.
        for idx in range(args.G):
            classifier_accuracy = compute_gated_accuracy(stored_metrics_classifier, idx)
            accuracy_tracker = net.module.accuracy_trackers[idx]
            if accuracy_tracker.should_freeze(classifier_accuracy):
                net.module.freeze_intermediate_classifier(idx)
                print(f"FREEZING CLASSIFIER {idx}")
            else:
                accuracy_tracker.insert_acc(classifier_accuracy)
        if use_mlflow:
            log_dict = log_metrics_mlflow(
                    prefix_logger='test',
                    gated_acc=gated_acc,
                    loss=loss,
                    G=args.G,
                    stored_per_x=stored_per_x_classifier,
                    stored_metrics=stored_metrics_classifier,
                    total_classifier=total,
                    batch=targets.size(0))

          
            gate_exit_acc = stored_metrics_gate['correct_exit_count'] / (total * args.G) * 100
            log_dict['test/gate_exit_acc'] = gate_exit_acc
            mlflow.log_metrics(log_dict,
                               step=batch_idx + (epoch * len(train_loader)))
    # Save checkpoint.
    if gated_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': gated_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(f'checkpoint_{args.dataset}_{args.model}'):
            os.mkdir(f'checkpoint_{args.dataset}_{args.model}')
        torch.save(
            state,
            f'./checkpoint_{args.dataset}_{args.model}/ckpt_{args.lr}_{args.wd}_{gated_acc}.pth'
        )
        best_acc = gated_acc
    if use_mlflow:
        log_dict = {'best/test_acc': gated_acc}
        mlflow.log_metrics(log_dict)
    return stored_metrics_classifier


# for epoch in range(start_epoch, start_epoch + args.num_epoch):
#     classifier_warmup_period = 0 if epoch > start_epoch else args.warmup_batch_count
#     stored_metrics_train = train(
#         epoch,
#         bilevel_opt=True,
#         bilevel_batch_count=args.bilevel_batch_count,
#         classifier_warmup_periods=classifier_warmup_period)
#     stored_metrics_test = test(epoch)
#     scheduler.step()

for epoch in range(start_epoch, start_epoch + args.num_epoch):
    train_boosted(epoch)
    # stored_metrics_test = test(epoch)
    scheduler.step()

mlflow.end_run()