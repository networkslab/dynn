'''Train DYNN from checkpoint of trained backbone'''

import os
import torch
import mlflow
from timm.models import *
from timm.models import create_model
from collect_metric_iter import aggregate_metrics, process_things
from data_loading.data_loader_helper import split_dataloader_in_n
from learning_helper import LearningHelper
from log_helper import log_aggregate_metrics_mlflow
from utils import  aggregate_dicts, progress_bar
from early_exit_utils import switch_training_phase
from models.t2t_vit import TrainingPhase
from plasticity_analysis.plasticity_metrics_utility import get_network_weight_norm_dict
import numpy as np

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

def train_single_epoch(args, helper: LearningHelper, device, train_loader, epoch, training_phase,
          bilevel_batch_count=20):
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
        metrics_of_batch = process_things(things_of_interest, gates_count=args.G,
                                          targets=targets, batch_size=batch_size,
                                          cost_per_exit=helper.net.module.normalized_cost_per_exit)
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
        
        if args.barely_train:
            if batch_idx > 20:
                print(
                    '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
                )
                return metrics_dict

    return metrics_dict



def evaluate(best_acc, args, helper: LearningHelper, device, init_loader, epoch, prefix_logger: str):
    helper.net.eval()
    metrics_dict = {}
    if prefix_logger == 'test': # we should split the data and combine at the end
        loaders = split_dataloader_in_n(init_loader, n=10)
    else:
        loaders = [init_loader]
    metrics_dicts = []
    log_dicts_of_trials = {}
    average_trials_log_dict = {}
    for loader in loaders:
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = targets.size(0)

            loss, things_of_interest = helper.get_surrogate_loss(inputs, targets)
            
            # obtain the metrics associated with the batch
            metrics_of_batch = process_things(things_of_interest, gates_count=args.G,
                                              targets=targets, batch_size=batch_size,
                                              cost_per_exit=helper.net.module.mult_add_at_exits)
            metrics_of_batch['loss'] = (loss.item(), batch_size)
            

            # keep track of the average metrics
            metrics_dict = aggregate_metrics(metrics_of_batch, metrics_dict, gates_count=args.G) 
            
            # format the metric ready to be displayed
            log_dict = log_aggregate_metrics_mlflow(
                    prefix_logger=prefix_logger,
                    metrics_dict=metrics_dict, gates_count=args.G) 
            display_progress_bar(prefix_logger=prefix_logger,training_phase=TrainingPhase.CLASSIFIER, step=batch_idx, total=len(loader), log_dict=log_dict)

            if args.barely_train:
                    if batch_idx > 50:
                        print(
                            '++++++++++++++WARNING++++++++++++++ you are barely testing to test some things'
                        )
                        break
        metrics_dicts.append(metrics_dict)
        for k, v in log_dict.items():
            aggregate_dicts(log_dicts_of_trials, k, v)
    for k,v in log_dicts_of_trials.items():
        average_trials_log_dict[k] = np.mean(v)
    
    gated_acc = average_trials_log_dict[prefix_logger+'/gated_acc']
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
        average_trials_log_dict[prefix_logger+'/test_acc']= gated_acc
        mlflow.log_metrics(average_trials_log_dict, step=epoch)
    return metrics_dict, best_acc, log_dicts_of_trials

# Any action based on the validation set
def set_from_validation(learning_helper, val_metrics_dict, freeze_classifier_with_val=False, alpha_conf = 0.04):
   
    # we fix the 1/0 ratios of of gate tasks based on the optimal percent exit in the validation sets
    
    exit_count_optimal_gate = val_metrics_dict['exit_count_optimal_gate'] # ({0: 0, 1: 0, 2: 0, 3: 0, 4: 6, 5: 72}, 128)
    total = exit_count_optimal_gate[1]
    pos_weights = []
    
    for gate, count in exit_count_optimal_gate[0].items():
        count = max(count, 0.1)
        pos_weight = total / count
        pos_weight = min(pos_weight, 5)
        pos_weights.append(pos_weight)
        
    
    learning_helper.gate_training_helper.set_ratios(pos_weights)
    

    ## compute the quantiles for the conformal intervals
    
    score, n = val_metrics_dict['gated_score']
    alpha_confs = [0.01,0.015,0.02,0.025,0.03,0.035,0.04,0.045,0.05]
    alpha_qhat_dict = {}
    for alpha_conf in alpha_confs:
        q_level = np.ceil((n+1)*(1-alpha_conf))/n
        qhat_general = np.quantile(score, q_level, method='higher')
        
        scores_per_gate, n = val_metrics_dict['score_per_gate']
        qhats = []
        for scores_in_l in scores_per_gate:
            if len(scores_in_l) > 10 :
                q_level = np.ceil((n+1)*(1-alpha_conf))/n
                qhat = np.quantile(scores_in_l, q_level, method='higher')
            else:
                qhat = qhat_general
            qhats.append(qhat)
        # add the last one
        final_score, n = val_metrics_dict['final_score']
        q_level = np.ceil((n+1)*(1-alpha_conf))/n
        qhat = np.quantile(final_score, q_level, method='higher')
        qhats.append(qhat)
        alpha_qhat_dict[alpha_conf] = {'qhats':qhats, 'qhat':qhat}

    learning_helper.classifier_training_helper.set_conf_thresholds(alpha_qhat_dict)
   

    #learning_helper.gate_training_helper.set_ratios(pos_weights)
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
        