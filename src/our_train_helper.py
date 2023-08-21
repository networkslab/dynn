'''Train DYNN from checkpoint of trained backbone'''

import os
import torch
import mlflow
from timm.models import *
from timm.models import create_model
from collect_metric_iter import aggregate_metrics, process_things
from learning_helper import LearningHelper
from log_helper import log_aggregate_metrics_mlflow
from utils import  progress_bar
from early_exit_utils import switch_training_phase
from models.t2t_vit import TrainingPhase
from plasticity_analysis.plasticity_metrics_utility import get_network_weight_norm_dict


def display_progress_bar(prefix_logger, training_phase, step, total, log_dict):
    loss = log_dict[prefix_logger+'/loss']
    if training_phase == TrainingPhase.WARMUP:
        progress_bar(step, total,'Loss: %.3f | Warmup  time' % (loss))
    elif training_phase == TrainingPhase.CLASSIFIER:
        gated_acc = log_dict[prefix_logger+'/gated_acc']
        progress_bar(
                step, total,
                'Classifier Loss: %.3f | Classifier Acc: %.3f%%' %
                (loss, gated_acc))
    elif training_phase == TrainingPhase.GATE:
        progress_bar(step, total, 'Gate Loss: %.3f ' % (loss)) 

def train_single_epoch(args, helper: LearningHelper, device, train_loader, epoch,training_phase,
          bilevel_batch_count=20,
          warmup_batch_count=0):
    print('\nEpoch: %d' % epoch)
    helper.net.train()

    metrics_dict = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)
       
        if training_phase == TrainingPhase.WARMUP:
            #  we compute the warmup loss
            loss, things_of_interest = helper.get_warmup_loss(inputs, targets)
        else:
            if batch_idx % bilevel_batch_count == 0:
                if helper.net.module.are_all_classifiers_frozen(): # no need to train classifiers anymore
                    training_phase = TrainingPhase.GATE
                    print("All classifiers are frozen, setting training phase to gate")
                else:
                    metrics_dict = {}
                    training_phase = switch_training_phase(training_phase)
            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets, training_phase)
        weight_norm_metrics = get_network_weight_norm_dict(helper.net.module)
        loss.backward()
        helper.optimizer.step()
        
        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gates_count=args.G, targets=targets, batch_size=batch_size) 
        metrics_of_batch['loss'] = (loss.item(), batch_size)
        
        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=args.G) 

        # format the metric ready to be displayed
        log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=args.G) 

        if args.use_mlflow:
            log_dict = log_dict | weight_norm_metrics
            mlflow.log_metrics(log_dict,
                                step=batch_idx +
                                (epoch * len(train_loader)))
        
        display_progress_bar('train', training_phase, step=batch_idx, total=len(train_loader), log_dict=log_dict)
        
        # if we are in warmup phase, we return if finished.
        if training_phase == TrainingPhase.WARMUP and batch_idx > warmup_batch_count:
            return metrics_dict
        if args.barely_train:
            if batch_idx > 50:
                print(
                    '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
                )
                return metrics_dict

    return metrics_dict



def test(best_acc, args, helper: LearningHelper, device, test_loader, epoch, freeze_classifier_with_val=False):
    helper.net.eval()
    metrics_dict = {}
    
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        batch_size = targets.size(0)

        loss, things_of_interest = helper.get_surrogate_loss(inputs, targets)
        # obtain the metrics associated with the batch
        metrics_of_batch = process_things(things_of_interest, gates_count=args.G, targets=targets, batch_size=batch_size) 
        metrics_of_batch['loss'] = (loss.item(), batch_size)
        
        # keep track of the average metrics
        metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=args.G) 
        
        # format the metric ready to be displayed
        log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='test',
                metrics_dict=metrics_dict, gates_count=args.G) 
        display_progress_bar(prefix_logger='test',training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(test_loader), log_dict=log_dict)

        if args.barely_train:
                if batch_idx > 50:
                    print(
                        '++++++++++++++WARNING++++++++++++++ you are barely testing to test some things'
                    )
                    break
        # TODO add metric for validation and early stopping
        # if freeze_classifier_with_val:
        #     for idx in range(args.G):
        #         classifier_accuracy = compute_gated_accuracy(stored_metrics_classifier, idx)
        #         accuracy_tracker = net.module.accuracy_trackers[idx]
        #         if accuracy_tracker.should_freeze(classifier_accuracy):
        #             net.module.freeze_intermediate_classifier(idx)
        #             print(f"FREEZING CLASSIFIER {idx}")
        #         else:
        #             accuracy_tracker.insert_acc(classifier_accuracy)
        

    gated_acc = log_dict['test/gated_acc']
    # Save checkpoint.
    if gated_acc > best_acc:
        print('Saving..')
        state = {
            'net': helper.net.state_dict(),
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
    if args.use_mlflow:
        log_dict['best/test_acc']= gated_acc
        mlflow.log_metrics(log_dict, step=epoch)
    return metrics_dict, best_acc


