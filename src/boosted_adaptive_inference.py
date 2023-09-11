from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import argparse
import numpy as np
import torch
import torch.nn as nn
import os
from torch import nn, optim
import mlflow
import torch.backends.cudnn as cudnn
import math
from log_helper import setup_mlflow
from data_loading.data_loader_helper import get_latest_checkpoint_path, get_cifar_10_dataloaders, get_cifar_100_dataloaders, get_path_to_project_root, split_dataloader_in_n
from timm.models import *
from timm.models import create_model
from metrics_utils import compute_detached_score, compute_detached_uncertainty_metrics

from models.register_models import *
from models.boosted_t2t_vit import Boosted_T2T_ViT
from utils import aggregate_dicts, free
import pickle as pk

class CustomizedOpen():
    def __init__(self, path, mode): 
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.f = open(self.path, self.mode)
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()


    
def stolen_calibrate(logits, targets, temp=None):
        

        if temp is None: # we compute the temp on this data
            # Stolen from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L78
            nll_criterion = nn.CrossEntropyLoss().cuda()
            #ece_criterion = _ECELoss().cuda()
            before_temperature_nll = nll_criterion(logits, targets.long()).item()
        # before_temperature_ece = ece_criterion(logits, labels).item()
           
            temp = nn.Parameter(torch.ones(1) * 1.5)
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([temp], lr=0.01, max_iter=50)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(logits/temp, targets.long())
                loss.backward()
                return loss
            optimizer.step(eval)

            # Calculate NLL and ECE after temperature scaling
            after_temperature_nll = nll_criterion(logits/temp, targets.long()).item()
           
            return temp.detach().cpu().numpy(), logits/temp
        else: # we calibrate the logits with the given temp
            return logits/temp

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)
     # split the validation into 2
    val_loader_1, val_loader_2 = split_dataloader_in_n(val_loader, n=2)
    n_test = 2
    test_loaders = split_dataloader_in_n(test_loader, n=n_test)
    # we find the threshold with validation set 1
    # we compute the quantiles for the conformal prediction with validation set 2
    val_pred_1, val_target_1 = tester.calc_logit(val_loader_1)
    val_pred_2, val_target_2 = tester.calc_logit(val_loader_2)
    test_preds = []
    test_targets = []
    for test_loader in test_loaders:
        test_pred, test_target = tester.calc_logit(test_loader)
        test_preds.append(test_pred)
        test_targets.append(test_target)
    
    number_of_layers = len(model.module.blocks)
    COST_PER_LAYER = 1.0/number_of_layers * 100
    costs_at_exit = [COST_PER_LAYER * (i + 1) for i in range(number_of_layers)]

    


     # split the validation into 2
    val_loader_1, val_loader_2 = split_dataloader_in_n(val_loader, n=2)
    n_test = 10
    test_loaders = split_dataloader_in_n(test_loader, n=n_test)
    # we find the threshold with validation set 1
    # we compute the quantiles for the conformal prediction with validation set 2
    val_pred_1, val_target_1 = tester.calc_logit(val_loader_1)
    val_pred_2, val_target_2 = tester.calc_logit(val_loader_2)
    test_preds = []
    test_targets = []
    for test_loader in test_loaders:
        test_pred, test_target = tester.calc_logit(test_loader)
        test_preds.append(test_pred)
        test_targets.append(test_target)
    
    

    acc_val_last = -1
    acc_test_last = -1
    path_project = get_path_to_project_root()
    save_path = os.path.join(path_project, args.result_dir, 'dynamic{}.txt'.format(args.save_suffix))
    all_value_dicts_per_T = []
    with CustomizedOpen(save_path, 'w') as fout:
        # for p in range(1, 100):
        for p in range(1, 40):
           # print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 15)
            n_blocks = len(model.module.blocks)
            probs = torch.exp(torch.log(_p) * torch.arange(1, n_blocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred_1, val_target_1, probs, costs_at_exit) # find the T with val_1
           
            metrics_dict_val = tester.dynamic_eval_with_threshold(val_pred_2, val_target_2, costs_at_exit, T) # compute conformal
            qhats_from_val = metrics_dict_val['qhats']
            temp_from_val = metrics_dict_val['temp']
            metrics_dicts = []
            for i in range(n_test):
                metrics_dict = tester.dynamic_eval_with_threshold(test_preds[i], test_targets[i], costs_at_exit, T, qhats=qhats_from_val, temp=temp_from_val
                )
                metrics_dicts.append(metrics_dict)
            
            mlflow_dict, all_value_dicts = get_ml_flow_dict(metrics_dicts)
            all_value_dicts_per_T.append(all_value_dicts)
            acc_test = mlflow_dict['ACC']
            exp_cost = mlflow_dict['EXPECTED_FLOPS']
            ece = mlflow_dict['ECE']
            
            #print('valid acc: {:.3f}, test acc: {:.3f}, test cost: {:.2f}, test ece: {:.2f}% '.format(acc_val, acc_test, exp_cost, metrics_dict['ECE']* 100.0))
            print('[{:.3f}, {:.2f}, {:.2f}],  '.format(acc_test, exp_cost, ece))
            mlflow.log_metrics(mlflow_dict, step=p)
            #fout.write('Cost: {}, Test acc {}\n'.format(exp_cost.item(), acc_test))
            # for k, v in metrics_dict.items():
            #     fout.write(f'{k}\n')
            #     for item in v:
            #         fout.write(f'{item}%\n')
            # fout.write('**************************\n')
    with open(args.dataset+'_boosted_results.pk', 'wb') as file:
        pk.dump(all_value_dicts_per_T, file)
        



def get_ml_flow_dict(dicts):
    all_value_dicts = {}
    for dict in dicts:
        
        for k, v in dict.items():
            if isinstance(v, list):
                for i in range(len(v)):
                    aggregate_dicts(all_value_dicts, f'{k}_{i}', v[i])
            else:
                aggregate_dicts(all_value_dicts, k, v)
            
    mlflow_dict = {}
    for key, vals in all_value_dicts.items():
        mlflow_dict[key] = np.mean(vals)
    return mlflow_dict, all_value_dicts

class Tester(object):
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.softmax = nn.Softmax(dim=1).cuda()

    def calc_logit(self, dataloader):
        self.model.eval()
        n_stage = len(self.model.module.blocks)
        logits = [[] for _ in range(n_stage)]
        targets = []
        for i, (input, target) in enumerate(dataloader):
            input = input.cuda()
            target = target.cuda()
            target = target.cpu()
            targets.append(target)
            with torch.no_grad():
                input_var = torch.autograd.Variable(input)
                output = self.model.forward(input_var)
                if not isinstance(output, list):
                    output = [output]
                for b in range(n_stage):
                    _t = self.softmax(output[b])
                    _t = _t.cpu()
                    logits[b].append(_t)

            if i % 50 == 0:
                print('Generate Logit: [{0}/{1}]'.format(i, len(dataloader)))

        for b in range(n_stage):
            logits[b] = torch.cat(logits[b], dim=0)

        size = (n_stage, logits[0].size(0), logits[0].size(1))
        ts_logits = torch.Tensor().resize_(size).zero_()
        for b in range(n_stage):
            ts_logits[b].copy_(logits[b])

        targets = torch.cat(targets, dim=0)
        ts_targets = torch.Tensor().resize_(size[1]).copy_(targets)

        return ts_logits, ts_targets

    def dynamic_eval_find_threshold(self, logits, targets, p, flops):
        """
            logits: m * n * c
            m: Stages
            n: Samples
            c: Classes
        """
        n_stage, n_sample, c = logits.size()

        max_preds, argmax_preds = logits.max(dim=2, keepdim=False)

        _, sorted_idx = max_preds.sort(dim=1, descending=True)

        filtered = torch.zeros(n_sample)
        T = torch.Tensor(n_stage).fill_(1e8)

        for k in range(n_stage - 1):
            acc, count = 0.0, 0
            out_n = math.floor(n_sample * p[k])
            for i in range(n_sample):
                ori_idx = sorted_idx[k][i]
                if filtered[ori_idx] == 0:
                    count += 1
                    if count == out_n:
                        T[k] = max_preds[k][ori_idx]
                        break
            filtered.add_(max_preds[k].ge(T[k]).type_as(filtered))

        T[n_stage -1] = -1e8 # accept all of the samples at the last stage

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force the sample to exit at k
                    if int(gold_label.item()) == int(argmax_preds[k][i].item()):
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all = 0
        for k in range(n_stage):
            _t = 1.0 * exp[k] / n_sample
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops, T

    def dynamic_eval_with_threshold(self, logits, targets, flops, thresholds, qhats=None, temp=None):
        metrics_dict = {}
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence
        gated_logits = torch.empty((logits.shape[1],logits.shape[2]))
        acc_rec, exit_count = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        correct_all = [0 for _ in range(n_stage)]
        acc_gated = []
        for i in range(n_sample):
            has_exited = False
            gold_label = targets[i]
            for k in range(n_stage):
                # compute acc over all samples, regardless of exit
                classifier_pred = int(argmax_preds[k][i].item())
                target = int(gold_label.item())
                if classifier_pred == target:
                    correct_all[k] += 1
                if max_preds[k][i].item() >= thresholds[k] and not has_exited: # exit at k
                    gated_logits[i] = logits[k,i]
                    if target == classifier_pred:
                        acc += 1
                        acc_rec[k] += 1
                    exit_count[k] += 1 # keeps track of number of exits per gate
                    has_exited = True # keep on looping but only for computing correct_all
        acc_all, sample_all = 0, 0
        if temp is None:
            temp, cal_logits = stolen_calibrate(gated_logits, targets)
        else:
            cal_logits = stolen_calibrate(gated_logits, targets, temp=temp)

        alphas = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]

        if qhats is None:
            qhats = []
            for alpha_conf in alphas:
                score = compute_detached_score(gated_logits, targets)
                q_level = np.ceil((n_sample+1)*(1-alpha_conf))/n_sample
                qhat = np.quantile(score, q_level, method='higher')
                qhats.append(qhat)
            metrics_dict['qhats'] = qhats
        
        elif qhats is not None:
            for i, qhat in enumerate(qhats):
                alpha = alphas[i]
                gated_prob = torch.softmax(gated_logits, dim=1)
                gated_prediction_sets = gated_prob >= (1-qhat)
                gated_average_inef = np.mean(np.sum(free(gated_prediction_sets.float()), axis=1))
                in_gated_conf = gated_prediction_sets[np.arange(n_sample),free(targets)]
                gated_average_cov = np.mean(free(in_gated_conf))
                metrics_dict['C_'+str(alpha)] = gated_average_inef
                metrics_dict['cov_'+str(alpha)] = gated_average_cov*100.0
        _, _, ece, _, _ = compute_detached_uncertainty_metrics(cal_logits, targets)
        for k in range(n_stage):
            _t = exit_count[k] * 1.0 / n_sample
            sample_all += exit_count[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]
        exit_rate = []
        for i in range(n_stage):
            acc_gated.append((acc_rec[i] / exit_count[i] * 100).item())
            correct_all[i] = correct_all[i] / n_sample * 100
            exit_rate.append(exit_count[i].item() / n_sample * 100)
        metrics_dict['GATED_ACC_PER_GATE'] = acc_gated
        metrics_dict['ALL_ACC_PER_GATE'] = correct_all
        metrics_dict['EXIT_RATE_PER_GATE'] = exit_rate
        acc = acc * 100.0 / n_sample
        metrics_dict['ACC'] = acc
        metrics_dict['EXPECTED_FLOPS'] = expected_flops.item()
        metrics_dict['ECE'] = ece*100.0
        metrics_dict['temp'] = temp
        return metrics_dict

def load_model_from_checkpoint(arch, checkpoint_path, device, num_classes, img_size):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    net = create_model(arch,
                       pretrained=False,
                       num_classes=num_classes,
                       drop_rate=0.0,
                       drop_connect_rate=None,
                       drop_path_rate=0.1,
                       drop_block_rate=None,
                       global_pool=None,
                       bn_tf=False,
                       bn_momentum=None,
                       bn_eps=None,
                       img_size=img_size)
    
    net.set_intermediate_heads(checkpoint['intermediate_head_positions'])
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    return net
def main(args):
    
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = vars(args)
    if args.use_mlflow:
        name = "boosted_adaptive_inference"
        setup_mlflow(name, cfg, experiment_name='boosted_evaluation')
   

    if args.dataset == 'cifar10':
        NUM_CLASSES = 10
        checkpoint_dir = "checkpoint_cifar10_t2t_7_boosted"
        _, val_loader, test_loader = get_cifar_10_dataloaders(img_size = IMG_SIZE, train_batch_size=64, test_batch_size=64, val_size=5000)
        num_classes = 10
    elif args.dataset == 'cifar100':
        NUM_CLASSES = 100
        checkpoint_dir = "checkpoint_cifar100_t2t_vit_14_boosted"
        _, val_loader, test_loader = get_cifar_100_dataloaders(img_size = IMG_SIZE, train_batch_size=64, test_batch_size=64, val_size=10000)
        num_classes = 100
    else:
        raise 'Unsupported dataset'
     # LOAD MODEL
    checkpoint_path = get_latest_checkpoint_path(checkpoint_dir)
    net = load_model_from_checkpoint(args.arch, checkpoint_path, device, NUM_CLASSES, IMG_SIZE)
    net = net.to(device)
    dynamic_evaluate(net, test_loader, val_loader, args)
    mlflow.end_run()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Boosted eval')
    parser.add_argument('--arch', type=str, choices=['t2t_vit_7_boosted', 't2t_vit_7', 't2t_vit_14_boosted'], default='t2t_vit_14_boosted', help='model')
    parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
    parser.add_argument('--result_dir', type=str, default="results",help='Directory for storing FLOP and acc')
    parser.add_argument('--use_mlflow',default=True,help='Store the run with mlflow')
    parser.add_argument('--base', type=int, default=4)
    parser.add_argument('--stepmode', type=str, choices=['even', 'lin_grow'])
    parser.add_argument('--step', type=int, default=1)
    parser.add_argument('--growthRate', type=int, default=6)
    parser.add_argument('--grFactor', default='1-2-4', type=str)
    parser.add_argument('--prune', default='max', choices=['min', 'max'])
    parser.add_argument('--bnFactor', default='1-2-4')
    parser.add_argument('--bottleneck', default=True, type=bool)
    parser.add_argument_group('boost', 'boosting setting')
    parser.add_argument('--lr_f', default=0.1, type=float, help='lr for weak learner')
    parser.add_argument('--lr_milestones', default='100,200', type=str, help='lr decay milestones')
    parser.add_argument('--ensemble_reweight', default="1.0", type=str, help='ensemble weight of early classifiers')
    parser.add_argument('--loss_equal', action='store_true', help='loss equalization')
    parser.add_argument('--save_suffix', default="t2t_vit_14_cifar_100",type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    main(args)