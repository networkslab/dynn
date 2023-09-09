import copy
import os
import torch
import mlflow
from timm.models import *
from timm.models import create_model
import pandas as pd
import time
from learning_helper import freeze_backbone as freeze_backbone_helper, LearningHelper
import math
from models.weighted import MetaSGD
import torch.nn.functional as F
from data_loading.data_loader_helper import get_abs_path

def train_weighted_net(train_loader, test_loader, model, meta_net, optimizer, meta_optimizer, num_epochs, args, with_serialization = False):
    num_exits = args.G + 1
    probs = calc_target_probs(num_exits)
    target_probs = probs[args.target_p_index-1]
    print(target_probs)
    best_acc_top_1 = 0
    for epoch in range(num_epochs):
        # note how the training takes the target_probs (from calc_probs(num_exits)[target_prob ==15)
        ce_loss_train, acc1_exits_train, _, meta_loss_train, lr, meta_lr, all_losses, all_confidences, all_weights = train_weighted_net_single_epoch(
            train_loader, model, meta_net, optimizer, meta_optimizer, epoch, target_probs, args)
        ce_loss_val, acc1_exits_val, _ = validate(test_loader, model, args)

        val_prec_last_head = acc1_exits_val[-1]
        if val_prec_last_head > best_acc_top_1:
            print("Increase in validation accuracy of last trainable head, serializing model")

            best_acc_top_1 = val_prec_last_head
            state = {
                'net': model.state_dict(),
                'acc': best_acc_top_1,
                'epoch': epoch,
            }
            checkpoint_path = get_abs_path(['checkpoint'])
            checkpoint_path = f'{checkpoint_path}/checkpoint_{args.dataset}_{args.arch}'
            if not os.path.isdir(checkpoint_path):
                os.mkdir(checkpoint_path)
            torch.save(
                state,
                f'{checkpoint_path}/ckpt_ep{epoch}_acc{best_acc_top_1}.pth'
            )



def train_weighted_net_single_epoch(train_loader, model, meta_net, optimizer, meta_optimizer, epoch, target_probs, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    meta_losses = AverageMeter('MetaLoss', ':.4e')
    top1, top5, all_losses = [], [], []
    num_exits = len(model.module.intermediate_heads)
    for i in range(num_exits):
        all_losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))
        top5.append(AverageMeter('Acc@5', ':6.2f'))

    # switch to train mode
    model.train()
    running_lr, running_meta_lr = None, None
    end = time.time()

    for i, (images, target) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        lr = adjust_learning_rate(optimizer, epoch, args, batch=i, nBatch=len(train_loader))
        meta_lr = adjust_meta_learning_rate(meta_optimizer, epoch, args, batch=i, nBatch=len(train_loader))

        if running_lr is None:
            running_lr = lr
        if running_meta_lr is None:
            running_meta_lr = meta_lr

        images_p1, images_p2 = images.chunk(2, dim=0)
        target_p1, target_p2 = target.chunk(2, dim=0)
        data_time.update(time.time() - end)


        ###################################################
        ## part 1: images_p1 as train, images_p2 as meta ##
        ###################################################

        if i % args.meta_interval == 0:
            # deep copy backbone

            pseudo_net = deep_copy_model(model, args)
            pseudo_net.train()

            pseudo_outputs = pseudo_net(images_p1)
            if not isinstance(pseudo_outputs, list):
                pseudo_outputs = [pseudo_outputs]

            # Here there was a branch in the original implementation based on args.meta_net_input_type, we use the default 'loss' path
            for j in range(num_exits):
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p1, reduction='none')
                if j==0:
                    pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                else:
                    pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
            input_of_meta = pseudo_losses

            pseudo_weight = meta_net(input_of_meta.detach())
            # Here there was a branch in the original implementation based on args.constraint_dimension, we use the default 'mat' path
            pseudo_weight = pseudo_weight - torch.mean(pseudo_weight)  # 1

            # this is 1 + ~w
            pseudo_weight = torch.ones(pseudo_weight.shape).to(pseudo_weight.device) + args.meta_weight_epsilon * pseudo_weight

            pseudo_loss_multi_exits = torch.sum(torch.mean(pseudo_weight * pseudo_losses, 0))
            pseudo_grads = torch.autograd.grad(
                pseudo_loss_multi_exits,
                pseudo_net.module.get_trainable_parameters(),
                create_graph=True
            )

            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            meta_outputs = pseudo_net(images_p2)
            if not isinstance(meta_outputs, list):
                meta_outputs = [meta_outputs]

            used_index = []
            meta_loss = 0.0
            for j in range(num_exits):
                with torch.no_grad():
                    confidence_target = F.softmax(meta_outputs[j], dim=1)
                    max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)
                    _, sorted_idx = max_preds_target.sort(dim=0, descending=True)
                    n_target = sorted_idx.shape[0]

                    if j == 0:
                        selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                        selected_index = selected_index.tolist()
                        used_index.extend(selected_index)
                    elif j < num_exits - 1:
                        filter_set = set(used_index)
                        unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                        selected_index = unused_index[: math.floor(n_target * target_probs[j])]
                        used_index.extend(selected_index)
                    else:
                        filter_set = set(used_index)
                        selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                if len(selected_index) > 0:
                    meta_loss += F.cross_entropy(meta_outputs[j][selected_index], target_p2[selected_index].long(), reduction='mean')
            meta_losses.update(meta_loss.item(), images_p2.size(0))

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        outputs = model(images_p1)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for j in range(num_exits):
            loss_vector = F.cross_entropy(outputs[j], target_p1, reduction='none')
            confidence = F.softmax(outputs[j], dim=1)
            confidence, _ = confidence.max(dim=1, keepdim=False)
            if j==0:
                losses = loss_vector.unsqueeze(1)
                confidences = confidence.unsqueeze(1)
            else:
                losses = torch.cat((losses, loss_vector.unsqueeze(1)), dim=1)
                confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)

        #  if args.meta_net_input_type == 'loss': use default as paper: feed the losses into the meta wpn
        input_of_meta = losses


        with torch.no_grad():
            weight = meta_net(input_of_meta)
            # Here there was a branch in the original implementation based on args.constraint_dimension, we use the default 'mat' path
            weight = weight - torch.mean(weight)
            weight = torch.ones(weight.shape).to(weight.device) + args.meta_weight_epsilon * weight
            if i == 0:
                all_losses_record = losses
                all_confidences_record = confidences
                all_weights = weight
            else:
                all_losses_record = torch.cat((all_losses_record, losses), dim=0)
                all_confidences_record = torch.cat((all_confidences_record, confidences), dim=0)
                all_weights = torch.cat((all_weights, weight), dim=0)

        loss_multi_exits = torch.mean(weight * losses, 0)

        for j in range(num_exits):
            all_losses[j].update(loss_multi_exits[j].item(), images_p1.size(0))
            prec1, prec5 = accuracy(outputs[j].data, target_p1, topk=(1, 5))
            top1[j].update(prec1.item(), images_p1.size(0))
            top5[j].update(prec5.item(), images_p1.size(0))

        loss_multi_exits = torch.sum(loss_multi_exits)

        optimizer.zero_grad()
        loss_multi_exits.backward()
        optimizer.step()

        # MUCH DUPLICATION TODO: Remove that shit into a method and just call it twice
        ###################################################
        ## part 2: images_p2 as train, images_p1 as meta ##
        ###################################################
        # tomorrow, this is where i left.
        if i % args.meta_interval == 0:
            # deep copy backbone
            pseudo_net = deep_copy_model(model, args)
            pseudo_net.train()

            pseudo_outputs = pseudo_net(images_p2)
            if not isinstance(pseudo_outputs, list):
                pseudo_outputs = [pseudo_outputs]


            for j in range(num_exits):
                pseudo_loss_vector = F.cross_entropy(pseudo_outputs[j], target_p2, reduction='none')
                if j==0:
                    pseudo_losses = pseudo_loss_vector.unsqueeze(1)
                else:
                    pseudo_losses = torch.cat((pseudo_losses, pseudo_loss_vector.unsqueeze(1)), dim=1)
            input_of_meta = pseudo_losses

            pseudo_weight = meta_net(input_of_meta.detach())  # TODO: .detach() or .data?

            pseudo_weight = pseudo_weight - torch.mean(pseudo_weight)  # 1

            pseudo_weight = torch.ones(pseudo_weight.shape).to(pseudo_weight.device) + args.meta_weight_epsilon * pseudo_weight

            pseudo_loss_multi_exits = torch.sum(torch.mean(pseudo_weight * pseudo_losses, 0))
            trainable_params = list(filter(lambda p: p.requires_grad, list(pseudo_net.parameters())))
            pseudo_grads = torch.autograd.grad(pseudo_loss_multi_exits, trainable_params, create_graph=True)

            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr=lr)
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grads)

            del pseudo_grads

            meta_outputs = pseudo_net(images_p1)
            if not isinstance(meta_outputs, list):
                meta_outputs = [meta_outputs]

            used_index = []
            meta_loss = 0.0
            for j in range(num_exits):
                with torch.no_grad():
                    confidence_target = F.softmax(meta_outputs[j], dim=1)
                    max_preds_target, _ = confidence_target.max(dim=1, keepdim=False)
                    _, sorted_idx = max_preds_target.sort(dim=0, descending=True)
                    n_target = sorted_idx.shape[0]

                    if j == 0:
                        selected_index = sorted_idx[: math.floor(n_target * target_probs[j])]
                        selected_index = selected_index.tolist()
                        used_index.extend(selected_index)
                    elif j < num_exits - 1:
                        filter_set = set(used_index)
                        unused_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                        selected_index = unused_index[: math.floor(n_target * target_probs[j])]
                        used_index.extend(selected_index)
                    else:
                        filter_set = set(used_index)
                        selected_index = [x.item() for x in sorted_idx if x.item() not in filter_set]
                if len(selected_index) > 0:
                    meta_loss += F.cross_entropy(meta_outputs[j][selected_index], target_p1[selected_index].long(), reduction='mean')
            meta_losses.update(meta_loss.item(), images_p1.size(0))

            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()

        outputs = model(images_p2)
        if not isinstance(outputs, list):
            outputs = [outputs]

        for j in range(num_exits):
            loss_vector = F.cross_entropy(outputs[j], target_p2, reduction='none')
            confidence = F.softmax(outputs[j], dim=1)
            confidence, _ = confidence.max(dim=1, keepdim=False)
            if j==0:
                losses = loss_vector.unsqueeze(1)
                confidences = confidence.unsqueeze(1)
            else:
                losses = torch.cat((losses, loss_vector.unsqueeze(1)), dim=1)
                confidences = torch.cat((confidences, confidence.unsqueeze(1)), dim=1)


        input_of_meta = losses


        with torch.no_grad():
            weight = meta_net(input_of_meta)
            weight = weight - torch.mean(weight) # 1
            weight = torch.ones(weight.shape).to(weight.device) + args.meta_weight_epsilon * weight
            if i == 0:
                all_losses_record = losses
                all_confidences_record = confidences
                all_weights = weight
            else:
                all_losses_record = torch.cat((all_losses_record, losses), dim=0)
                all_confidences_record = torch.cat((all_confidences_record, confidences), dim=0)
                all_weights = torch.cat((all_weights, weight), dim=0)

        loss_multi_exits = torch.mean(weight * losses, 0)

        for j in range(num_exits):
            all_losses[j].update(loss_multi_exits[j].item(), images_p2.size(0))
            prec1, prec5 = accuracy(outputs[j].data, target_p2, topk=(1, 5))
            top1[j].update(prec1.item(), images_p2.size(0))
            top5[j].update(prec5.item(), images_p2.size(0))

        loss_multi_exits = torch.sum(loss_multi_exits)

        optimizer.zero_grad()
        loss_multi_exits.backward()
        optimizer.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i % 40 == 0):
            print('Epoch: [{0}][{1}/{2}]\t'
                              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                              'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                              'MetaLoss {meta_loss.val:.4f} ({meta_loss.avg:.4f})\t'
                              'Acc@1 {top1.val:.4f} ({top1.avg:.4f})\t'
                              'Acc@5 {top5.val:.4f} ({top5.avg:.4f})'.format(
                epoch, i + 1, len(train_loader),
                batch_time=batch_time, data_time=data_time,
                loss=all_losses[-1], meta_loss=meta_losses, top1=top1[-1], top5=top5[-1]))

    ce_loss = []
    acc1_exits = []
    acc5_exits = []
    for j in range(num_exits):
        ce_loss.append(all_losses[j].avg)
        acc1_exits.append(top1[j].avg)
        acc5_exits.append(top5[j].avg)

    return ce_loss, acc1_exits, acc5_exits, meta_losses.avg, running_lr, running_meta_lr, all_losses_record, all_confidences_record, all_weights


def validate(loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    criterion = torch.nn.CrossEntropyLoss()
    data_time = AverageMeter('Data', ':6.3f')
    model.eval()
    num_exits = args.G
    top1, top5, losses = [], [], []
    for i in range(num_exits):
        losses.append(AverageMeter('Loss', ':.4e'))
        top1.append(AverageMeter('Acc@1', ':6.2f'))
        top5.append(AverageMeter('Acc@5', ':6.2f'))

    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(loader):
            data_time.update(time.time() - end)

            target = target.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)

            output = model(images)
            if not isinstance(output, list):
                output = [output]

            for j in range(num_exits):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                loss = criterion(output[j], target)

                top1[j].update(prec1.item(), images.size(0))
                top5[j].update(prec5.item(), images.size(0))
                losses[j].update(loss.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % 40 == 0:
                print('Epoch: [{0}/{1}]\t'
                                  'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'
                                  'Data {data_time.val:.3f}({data_time.avg:.3f})\t'
                                  'Acc@1 {top1.val:.4f}({top1.avg:.4f})\t'
                                  'Acc@5 {top5.val:.4f}({top5.avg:.4f})'
                .format(
                    i, len(loader),
                    batch_time=batch_time, data_time=data_time,
                    top1=top1[-1], top5=top5[-1]))

        ce_loss = []
        acc1_exits = []
        acc5_exits = []
        for j in range(num_exits):
            ce_loss.append(losses[j].avg)
            acc1_exits.append(top1[j].avg)
            acc5_exits.append(top5[j].avg)

        df = pd.DataFrame({'ce_loss': ce_loss, 'acc1_exits': acc1_exits, 'acc5_exits':acc5_exits})

        log_file = f"{get_abs_path(['results'])}/weighted_val_{args.dataset}_{args.arch}.csv"
        with open(log_file, "w") as f:
            df.to_csv(f)

        return ce_loss, acc1_exits, acc5_exits

# Utility classes and methods
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def adjust_learning_rate(optimizer, epoch, args, batch=None, nBatch=None):
    T_total = args.num_epoch * nBatch
    T_cur = (epoch % args.num_epoch) * nBatch + batch
    lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_meta_learning_rate(meta_optimizer, epoch, args, batch=None, nBatch=None):
    T_total = args.num_epoch * nBatch
    T_cur = (epoch % args.num_epoch) * nBatch + batch
    meta_lr = 0.5 * args.meta_lr * (1 + math.cos(math.pi * T_cur / T_total))
    for param_group in meta_optimizer.param_groups:
        param_group['lr'] = meta_lr
    return meta_lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def calc_target_probs(num_exits):
    for p in range(1, 40):
        _p = torch.FloatTensor(1).fill_(p * 1.0 / 20)
        probs = torch.exp(torch.log(_p) * torch.arange(1, num_exits + 1))
        probs /= probs.sum()
        if p == 1:
            probs_list = probs.unsqueeze(0)
        else:
            probs_list = torch.cat((probs_list, probs.unsqueeze(0)), 0)

    return probs_list

def deep_copy_model(model, args):
    NUM_CLASSES = 10
    IMG_SIZE = 224
    model_name = args.arch
    pseudo_net = create_model(model_name, # TODO configure this to accept the architecture (boosted vs others etc...)
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
    transformer_layer_gating = [g for g in range(args.G)]
    pseudo_net.set_intermediate_heads(transformer_layer_gating)

    device = next(model.parameters()).device
    pseudo_net.load_state_dict(model.module.state_dict()) # copy the state of the current model in a separate model
    pseudo_net = pseudo_net.to(device) # check this works
    if 'cuda' in str(device):
        pseudo_net = torch.nn.DataParallel(pseudo_net)
    unfrozen_modules = ['intermediate_heads']
    freeze_backbone_helper(pseudo_net, unfrozen_modules)
    return pseudo_net