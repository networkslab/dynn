from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
import argparse

import torch
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
import math
from data_loading.data_loader_helper import get_latest_checkpoint_path, get_cifar_10_dataloaders
from timm.models import *
from timm.models import create_model

from models.t2t_vit import Boosted_T2T_ViT


class CustomizedOpen():
    def __init__(self, path, mode):
        self.path = path
        self.mode = mode

    def __enter__(self):
        self.f = open(self.path, self.mode)
        return self.f

    def __exit__(self, type, value, traceback):
        self.f.close()

def dynamic_evaluate(model, test_loader, val_loader, args):
    tester = Tester(model, args)

    val_pred, val_target = tester.calc_logit(val_loader)
    test_pred, test_target = tester.calc_logit(test_loader)

    COST_PER_LAYER = 1.0/7 * 100
    costs_at_exit = [COST_PER_LAYER * (i + 1) for i in range(len(model.module.blocks))]

    acc_val_last = -1
    acc_test_last = -1
    save_path = os.path.join(args.result_dir, 'dynamic{}.txt'.format(args.save_suffix))

    with CustomizedOpen(save_path, 'w') as fout:
        # for p in range(1, 100):
        for p in range(1, 40):
            print("*********************")
            _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
            n_blocks = len(model.module.blocks)
            probs = torch.exp(torch.log(_p) * torch.range(1, n_blocks))
            probs /= probs.sum()
            acc_val, _, T = tester.dynamic_eval_find_threshold(val_pred, val_target, probs, costs_at_exit)
            acc_test, exp_cost = tester.dynamic_eval_with_threshold(test_pred, test_target, costs_at_exit, T)
            print('valid acc: {:.3f}, test acc: {:.3f}, test cost: {:.2f}%'.format(acc_val, acc_test, exp_cost))
            fout.write('{} {}\n'.format(exp_cost.item(), acc_test))


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

    def dynamic_eval_with_threshold(self, logits, targets, flops, T):
        n_stage, n_sample, _ = logits.size()
        max_preds, argmax_preds = logits.max(dim=2, keepdim=False) # take the max logits as confidence

        acc_rec, exp = torch.zeros(n_stage), torch.zeros(n_stage)
        acc, expected_flops = 0, 0
        for i in range(n_sample):
            gold_label = targets[i]
            for k in range(n_stage):
                if max_preds[k][i].item() >= T[k]: # force to exit at k
                    _g = int(gold_label.item())
                    _pred = int(argmax_preds[k][i].item())
                    if _g == _pred:
                        acc += 1
                        acc_rec[k] += 1
                    exp[k] += 1
                    break
        acc_all, sample_all = 0, 0
        for k in range(n_stage):
            _t = exp[k] * 1.0 / n_sample
            sample_all += exp[k]
            expected_flops += _t * flops[k]
            acc_all += acc_rec[k]

        return acc * 100.0 / n_sample, expected_flops

def load_model_from_checkpoint(checkpoint_path, device, num_classes, img_size):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    
    net = create_model('t2t_vit_7_boosted',
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
    NUM_CLASSES = 10
    IMG_SIZE = 224
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # LOAD MODEL
    checkpoint_path = get_latest_checkpoint_path('checkpoint_cifar10_t2t_7_boosted')
    net = load_model_from_checkpoint(checkpoint_path, device, NUM_CLASSES, IMG_SIZE)
    net = net.to(device)
    
    _, val_loader, test_loader = get_cifar_10_dataloaders(img_size = IMG_SIZE, train_batch_size=64, test_batch_size=64, val_size=5000)
    dynamic_evaluate(net, test_loader, val_loader, args)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Boosted eval')
    parser.add_argument('--nBlocks', type=int, default=1)
    parser.add_argument('--nChannels', type=int, default=32)
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
    parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
    parser.add_argument('--save_suffix', type=str)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--result_dir', type=str, help='Directory for storing FLOP and acc')
    args = parser.parse_args()
    main(args)