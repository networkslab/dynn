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
from boosted_training_helper import test_boosted, train_boosted
from data_loading.data_loader_helper import get_abs_path, get_cifar_10_dataloaders, get_path_to_project_root, get_cifar_100_dataloaders
from learning_helper import freeze_backbone as freeze_backbone_helper, LearningHelper
from log_helper import setup_mlflow
from models.classifier_training_helper import LossContributionMode
from models.custom_modules.gate import GateType
from models.gate_training_helper import GateObjective
from our_train_helper import set_from_validation, evaluate, train_single_epoch
from threshold_helper import fixed_threshold_test
from utils import fix_the_seed
from models.register_models import *
from models.boosted_t2t_vit import Boosted_T2T_ViT
from models.t2t_vit import GateTrainingScheme, GateSelectionMode, TrainingPhase

parser = argparse.ArgumentParser(
    description='PyTorch CIFAR10/CIFAR100 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--arch', type=str,
                    choices=['t2t_vit_7_boosted', 't2t_vit_7_baseline','t2t_vit_7', 't2t_vit_14', 't2t_vit_14_boosted'], # baseline is to train only with warmup, no gating
                    default='t2t_vit_7', help='model to train'
                    )
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--min-lr',default=2e-4,type=float,help='minimal learning rate')
parser.add_argument('--dataset',type=str,default='cifar10',help='cifar10 or cifar100')
parser.add_argument('--batch', type=int, default=64, help='batch size')
parser.add_argument('--ce_ic_tradeoff',default=0.1,type=float,help='cost inference and cross entropy loss tradeoff')
parser.add_argument('--G', default=8, type=int, help='number of gates')
parser.add_argument('--num_epoch', default=5, type=int, help='num of epochs')
parser.add_argument('--bilevel_batch_count',default=200,type=int,help='number of batches before switching the training modes')
parser.add_argument('--barely_train',action='store_true',help='not a real run')
parser.add_argument('--resume', '-r',action='store_true',help='resume from checkpoint')
parser.add_argument('--model', type=str,default='learn_gate_direct')  # learn_gate, learn_gate_direct
parser.add_argument('--gate',type=GateType,default=GateType.UNCERTAINTY,choices=GateType)  # unc, code, code_and_unc
parser.add_argument('--drop-path',type=float,default=0.1,metavar='PCT',help='Drop path rate (default: None)')
parser.add_argument('--gate_selection_mode', type=GateSelectionMode, default=GateSelectionMode.DETERMINISTIC, choices=GateSelectionMode)
parser.add_argument('--gate_objective', type=GateObjective, default=GateObjective.CrossEntropy, choices=GateObjective)
parser.add_argument('--transfer-ratio',type=float,default=0.01, help='lr ratio between classifier and backbone in transfer learning')
parser.add_argument('--gate_training_scheme',default='EXIT_SUBSEQUENT', help='Gate training scheme (how to handle gates after first exit)',
    choices=['DEFAULT', 'IGNORE_SUBSEQUENT', 'EXIT_SUBSEQUENT'])
parser.add_argument('--proj_dim',default=32,help='Target dimension of random projection for ReLU codes')
parser.add_argument('--num_proj',default=16,help='Target number of random projection for ReLU codes')
parser.add_argument('--use_mlflow',default=True, help='Store the run with mlflow')
parser.add_argument('--classifier_loss', type=LossContributionMode, default=LossContributionMode.BOOSTED, choices=LossContributionMode)
args = parser.parse_args()

fix_the_seed(seed=322)

if args.barely_train:
    print(
        '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
    )
gate_training_scheme = GateTrainingScheme[args.gate_training_scheme]

if args.use_mlflow:
    if 'boosted' in args.arch:
        name = 'boosted'
    elif 'baseline' in args.arch:
        name = 'baseline'
    else:
        name = "_".join([ str(a) for a in [args.ce_ic_tradeoff, args.classifier_loss]])
    cfg = vars(args)
    if args.barely_train:
        setup_mlflow(name, cfg, experiment_name='test run')
    else:
        setup_mlflow(name, cfg, experiment_name='conformal')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
path_project = get_path_to_project_root()
model = args.arch

if args.dataset=='cifar10':
    NUM_CLASSES = 10
    IMG_SIZE = 224
    args.G = 6
    train_loader, val_loader, test_loader  = get_cifar_10_dataloaders(img_size = IMG_SIZE,train_batch_size=args.batch, 
                                                    test_batch_size=args.batch, val_size=5000)
    checkpoint = torch.load(os.path.join(path_project, 'checkpoint/checkpoint_cifar10_t2t_vit_7/ckpt_0.01_0.0005_94.95.pth'),
                        map_location=torch.device(device))
elif args.dataset=='cifar100':
    NUM_CLASSES = 100
    IMG_SIZE = 224
    args.G = 13
    train_loader, val_loader, test_loader = get_cifar_100_dataloaders(img_size = IMG_SIZE,train_batch_size=args.batch, val_size=10000)
    checkpoint = torch.load(os.path.join(path_project, 'checkpoint/cifar100_t2t-vit-14_88.4.pth'),
                        map_location=torch.device(device))
transformer_layer_gating = [g for g in range(args.G)]
print(f'learning rate:{args.lr}, weight decay: {args.wd}')
# create T2T-ViT Model
print('==> Building model..')
net = create_model(model, # TODO configure this to accept the architecture (boosted vs others etc...)
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

if isinstance(net.module, Boosted_T2T_ViT):
    for epoch in range(0, args.num_epoch):
        train_boosted(args, net, device, train_loader, optimizer, epoch)
        accs = test_boosted(args, net, test_loader, epoch)
        # stored_metrics_test = test(epoch)
        scheduler.step()
    state = {
        'state_dict': net.state_dict(),
        'intermediate_head_positions': net.module.intermediate_head_positions
    }
    checkpoint_folder_path = get_abs_path(["checkpoint"])

    # WARNING: for 7 boosted the checkpoint was saved as f'{checkpoint_folder_path}/checkpoint_{args.dataset}_t2t_7_boosted'
    target_checkpoint_folder_path = f'{checkpoint_folder_path}/checkpoint_{args.dataset}_{args.arch}'
    if not os.path.isdir(target_checkpoint_folder_path):
        os.mkdir(target_checkpoint_folder_path)
    torch.save(state, f'{target_checkpoint_folder_path}/last_{accs[-1]}_second_to_last_{accs[-2]}.pth')

elif 'baseline' in args.arch: # only training with warmup
    learning_helper = LearningHelper(net, optimizer, args, device)
    
    for epoch in range(0, args.num_epoch):
        train_single_epoch(args, learning_helper, device, train_loader, epoch=epoch, training_phase=TrainingPhase.WARMUP, bilevel_batch_count=args.bilevel_batch_count)
        #stored_metrics_test = evaluate(best_acc, args, learning_helper, device, test_loader, epoch)
        fixed_threshold_test(args,learning_helper, device, test_loader, val_loader)
        scheduler.step()

else:
    # start with warm up for the first epoch
    learning_helper = LearningHelper(net, optimizer, args, device)
    train_single_epoch(args, learning_helper, device, train_loader, epoch=0, training_phase=TrainingPhase.WARMUP, bilevel_batch_count=args.bilevel_batch_count)
    val_metrics_dict, _ = evaluate(best_acc, args, learning_helper, device, val_loader, epoch=0, prefix_logger='val')
    set_from_validation(learning_helper, val_metrics_dict)
    evaluate(best_acc, args, learning_helper, device, test_loader, epoch=0, prefix_logger='test')
    for epoch in range(1, args.num_epoch):
        train_single_epoch(args, learning_helper, device, train_loader, epoch=epoch, training_phase=TrainingPhase.CLASSIFIER, bilevel_batch_count=args.bilevel_batch_count)
        val_metrics_dict, _ = evaluate(best_acc, args, learning_helper, device, val_loader, epoch, prefix_logger='val')
        evaluate(best_acc, args, learning_helper, device, test_loader, epoch, prefix_logger='test')
        set_from_validation(learning_helper, val_metrics_dict)
        #fixed_threshold_test(args,learning_helper, device, test_loader, val_loader) # this can make gpu run OOM
        scheduler.step()
mlflow.end_run()