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
from learning_helper import get_loss, get_surrogate_loss, get_weighted_loss, get_boosted_loss, freeze_backbone as freeze_backbone_helper
from log_helper import log_metrics_mlflow, setup_mlflow, compute_gated_accuracy
from models.custom_modules.gate import GateType
from utils import fix_the_seed, progress_bar
from early_exit_utils import switch_training_phase
from models.t2t_vit import TrainingPhase, GateTrainingScheme, GateSelectionMode, Boosted_T2T_ViT

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--arch', type=str,
                    choices=['t2t_vit_7_boosted', 't2t_vit_7', 't2t_vit_14'],
                    default='t2t_vit_7', help='model to train'
                    )
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr',default=2e-4,type=float,help='minimal learning rate')
parser.add_argument('--dataset',type=str,default='cifar10',help='cifar10 or cifar100')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--ce_ic_tradeoff',default=1.5,type=float,help='cost inference and cross entropy loss tradeoff')
parser.add_argument('--G', default=6, type=int, help='number of gates')
parser.add_argument('--num_epoch', default=8, type=int, help='num of epochs')
parser.add_argument('--warmup_batch_count',default=700,type=int,help='number of batches for warmup where all classifier are trained')
parser.add_argument('--bilevel_batch_count',default=200,type=int,help='number of batches before switching the training modes')
parser.add_argument('--barely_train',action='store_true',help='not a real run')
parser.add_argument('--resume','-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--model', type=str,default='learn_gate_direct')  # learn_gate, learn_gate_direct
parser.add_argument('--gate',type=GateType,default=GateType.CODE_AND_UNC,choices=GateType)  # unc, code, code_and_unc
parser.add_argument('--drop-path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: None)')
parser.add_argument('--gate_selection_mode', type=GateSelectionMode, default=GateSelectionMode.DETERMINISTIC, choices=GateSelectionMode)
parser.add_argument('--transfer-ratio',type=float,default=0.01, help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--gate_training_scheme',default='EXIT_SUBSEQUENT',help='Gate training scheme (how to handle gates after first exit)',
    choices=['DEFAULT', 'IGNORE_SUBSEQUENT', 'EXIT_SUBSEQUENT'])
parser.add_argument('--proj_dim',default=32,help='Target dimension of random projection for ReLU codes')
parser.add_argument('--num_proj',default=16,help='Target number of random projection for ReLU codes')
parser.add_argument('--use_mlflow',default=True,help='Store the run with mlflow')
parser.add_argument('--weighted_class_loss', default=True, help='How to compute loss of classifiers')
args = parser.parse_args()


fix_the_seed(seed=322)
weighted = args.weighted_class_loss != 'False'
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
            args.gate_training_scheme, f'{"WEIGHTED" if weighted else "SURR"}'
        ]
    ])
    cfg = vars(args)
    setup_mlflow(name, cfg)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
path_project = get_path_to_project_root()
model = args.arch

if args.dataset=='cifar10':
    NUM_CLASSES = 10
    IMG_SIZE = 224
    args.G = 6
    train_loader, test_loader = get_cifar_10_dataloaders(img_size = IMG_SIZE,train_batch_size=args.batch, test_batch_size=args.batch)
    checkpoint = torch.load(os.path.join(path_project, 'checkpoint/checkpoint_cifar10_t2t_vit_7/ckpt_0.01_0.0005_94.95.pth'),
                        map_location=torch.device(device))
elif args.dataset=='cifar100':
    NUM_CLASSES = 100
    IMG_SIZE = 224
    args.G = 13
    train_loader, test_loader = get_cifar_100_dataloaders(img_size = IMG_SIZE,train_batch_size=args.batch)
    checkpoint = torch.load(os.path.join(path_project, 'checkpoint/cifar100_t2t-vit-14_88.4.pth'),
                        map_location=torch.device(device))
transformer_layer_gating = [g for g in range(args.G)]
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model(model,  # TODO configure this to accept the architecture (boosted vs others etc...)
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
print(args.G)
direct_exit_prob_param = args.model == 'learn_gate_direct'
if not isinstance(net, Boosted_T2T_ViT):
    net.set_learnable_gates(device,
                            transformer_layer_gating,
                            direct_exit_prob_param=direct_exit_prob_param,
                            gate_type=args.gate,
                            proj_dim=int(args.proj_dim),
                            num_proj=int(args.num_proj))

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print('==> Resuming from checkpoint..')
checkpoint_path = os.path.join(get_path_to_project_root(), 'checkpoint')
assert os.path.isdir(checkpoint_path)
param_with_issues = net.load_state_dict(checkpoint['net'], strict=False)
print("Missing keys:", param_with_issues.missing_keys)
print("Unexpected keys:", param_with_issues.unexpected_keys)
best_acc = checkpoint['acc']
start_epoch = checkpoint['epoch']

# Backbone is always frozen
unfrozen_modules = ['intermediate_heads', 'gates'] if not isinstance(net.module, Boosted_T2T_ViT) else ['intermediate_heads']
freeze_backbone_helper(net, unfrozen_modules)
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
            if weighted:
                loss, things_of_interest = get_weighted_loss(
                    inputs, targets, optimizer, net, training_phase=training_phase)
            else:
                loss, things_of_interest = get_surrogate_loss(inputs, targets, optimizer, net, training_phase=training_phase)
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


def test_boosted(epoch):
    net.eval()
    n_blocks = len(net.module.blocks)
    corrects = [0] * n_blocks
    totals = [0] * n_blocks
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            outs = net.module.forward(x)
        for i, out in enumerate(outs):
            corrects[i] += (torch.argmax(out, 1) == y).sum().item()
            totals[i] += y.shape[0]
    corrects = [c / t * 100 for c, t in zip(corrects, totals)]
    log_dict = {}
    for blk in range(n_blocks):
        log_dict['test' + '/accuracies' +
                 str(blk)] = corrects[blk]
    mlflow.log_metrics(log_dict)
    return corrects

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
        # for idx in range(args.G):
        #     classifier_accuracy = compute_gated_accuracy(stored_metrics_classifier, idx)
        #     accuracy_tracker = net.module.accuracy_trackers[idx]
        #     if accuracy_tracker.should_freeze(classifier_accuracy):
        #         net.module.freeze_intermediate_classifier(idx)
        #         print(f"FREEZING CLASSIFIER {idx}")
        #     else:
        #         accuracy_tracker.insert_acc(classifier_accuracy)
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


if isinstance(net.module, Boosted_T2T_ViT):
    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        train_boosted(epoch)
        accs = test_boosted(epoch)
        # stored_metrics_test = test(epoch)
        scheduler.step()
    state = {
        'state_dict': net.state_dict(),
        'intermediate_head_positions': net.module.intermediate_head_positions
    }
    checkpoint_folder_path = get_abs_path(["checkpoint"])
    target_checkpoint_folder_path = f'{checkpoint_folder_path}/checkpoint_{args.dataset}_t2t_7_boosted'
    if not os.path.isdir(target_checkpoint_folder_path):
        os.mkdir(target_checkpoint_folder_path)
    torch.save(state, f'{target_checkpoint_folder_path}/ckpt_7_{accs[-1]}_6_{accs[-2]}.pth')
else:
    for epoch in range(start_epoch, start_epoch + args.num_epoch):
        classifier_warmup_period = 0 if epoch > start_epoch else args.warmup_batch_count
        stored_metrics_train = train(
            epoch,
            bilevel_opt=True,
            bilevel_batch_count=args.bilevel_batch_count,
            classifier_warmup_periods=classifier_warmup_period)
        stored_metrics_test = test(epoch)
        scheduler.step()

mlflow.end_run()