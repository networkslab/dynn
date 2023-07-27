'''Train DYNN from checkpoint of trained backbone'''

import os
import torch
import mlflow
from timm.models import *
from timm.models import create_model
from collect_metric_iter import aggregate_metrics, process_things
from learning_helper import get_warmup_loss, get_surrogate_loss, get_weighted_loss
from log_helper import log_aggregate_metrics_mlflow
from utils import  progress_bar
from early_exit_utils import switch_training_phase
from models.t2t_vit import TrainingPhase


def warmup_train(args, net, device, train_loader, optimizer, epoch,warmup_batch_count=0):
    print('Warming up Epoch: %d' % epoch)
    cumul_data = 0
    cumul_loss = 0
    metrics_dict = {}
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        #  we compute the warmup loss
        loss, things_of_interest = get_warmup_loss(inputs, targets, optimizer,net)
        
        batch_size = targets.size(0)
        cumul_data += batch_size
        loss.backward()
        optimizer.step()
        
        metrics_to_aggregate_dict = process_things(things_of_interest, gates_count=args.G, targets=targets, batch_size=batch_size)
        metrics_dict = aggregate_metrics(metrics_to_aggregate_dict, metrics_dict,gates_count=args.G)
        
        
        cumul_loss += loss.item()
        loss = cumul_loss / cumul_data
        progress_bar(batch_idx, len(train_loader),
                        'Loss: %.3f | Warmup  time' % (loss))

        if args.use_mlflow:
            log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=args.G)

            mlflow.log_metrics(log_dict,
                                step=batch_idx +
                                (epoch * len(train_loader)))
        if batch_idx> warmup_batch_count:
            break
        if args.barely_train:
            if batch_idx > 50:
                print(
                    '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
                )
                break

    return metrics_dict

def train(args, net, device, train_loader, optimizer, epoch,
          bilevel_opt=False,
          bilevel_batch_count=20):
    print('\nEpoch: %d' % epoch)
    net.train()
    cumul_data = 0
    cumul_loss = 0
    metrics_dict = {}
    training_phase = TrainingPhase.GATE
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        
        if bilevel_opt and batch_idx % bilevel_batch_count == 0:
            if net.module.are_all_classifiers_frozen(): # no need to train classifiers anymore
                training_phase = TrainingPhase.GATE
                print("All classifiers are frozen, setting training phase to gate")
            else:
                training_phase = switch_training_phase(training_phase)

    
        if args.weighted_class_loss:
            loss, things_of_interest = get_weighted_loss(
                inputs, targets, optimizer, net, training_phase=training_phase)
        else:
            loss, things_of_interest = get_surrogate_loss(inputs, targets, optimizer, net, training_phase=training_phase)
       
        batch_size = targets.size(0)
        cumul_data += batch_size
        loss.backward()
        optimizer.step()
        
        metrics_to_aggregate_dict = process_things(things_of_interest, gates_count=args.G, targets=targets, batch_size=batch_size)
        metrics_dict = aggregate_metrics(metrics_to_aggregate_dict, metrics_dict,gates_count=args.G)
        
        
        cumul_loss += loss.item()
        loss = cumul_loss / cumul_data
        progress_bar(batch_idx, len(train_loader),
                        'Loss: %.3f | Warmup  time' % (loss))

        if args.use_mlflow:
            log_dict = log_aggregate_metrics_mlflow(
                prefix_logger='train',
                metrics_dict=metrics_dict, gates_count=args.G)

            mlflow.log_metrics(log_dict,
                                step=batch_idx +
                                (epoch * len(train_loader)))

        # if training_phase == TrainingPhase.CLASSIFIER:
        #     gated_y_logits = things_of_interest['gated_y_logits']
        #     _, predicted = gated_y_logits.max(1)
        #     correct += predicted.eq(targets).sum().item()
        #     total_classifier += targets.size(0)
        #     # compute metrics to display
        #     gated_acc = 100. * correct / total_classifier
        #     classifier_loss += loss.item()
        #     loss = classifier_loss / total_classifier
        #     progress_bar(
        #         batch_idx, len(train_loader),
        #         'Classifier Loss: %.3f | Classifier Acc: %.3f%% (%d/%d)' %
        #         (loss, gated_acc, correct, total_classifier))

        #     stored_metrics['acc'] = gated_acc
        #     if args.use_mlflow:
        #         log_dict = log_metrics_mlflow(
        #             prefix_logger='train',
        #             gated_acc=gated_acc,
        #             loss=loss,
        #             G=args.G,
        #             stored_per_x=stored_per_x,
        #             stored_metrics=stored_metrics,
        #             total_classifier=total_classifier,
        #             batch=targets.size(0))

        #         mlflow.log_metrics(log_dict,
        #                            step=batch_idx +
        #                            (epoch * len(train_loader)))
        

        # elif training_phase == TrainingPhase.GATE:
        #     total_gate += targets.size(0)
        #     gate_loss += loss.item()
        #     loss = gate_loss / total_gate
        #     progress_bar(batch_idx, len(train_loader), 'Gate Loss: %.3f ' % (loss))
        #     exit_count_optimal_gate_perc = {k: v / total_gate * 100 for k, v in stored_metrics['exit_count_optimal_gate'].items()}
        #     gate_exit_acc = stored_metrics['correct_exit_count'] / (total_gate * args.G) * 100
        #     log_dict ={'train/gate_loss':loss, 'train/gate_exit_acc': gate_exit_acc}
        #     for g in range(args.G):
        #         log_dict['train' + '/optimal_percent_exit' +
        #                  str(g)] = exit_count_optimal_gate_perc[g]
        #     mlflow.log_metrics(log_dict,
        #                        step=batch_idx + (epoch * len(train_loader)))
        # if args.barely_train:
        #     if batch_idx > 50:
        #         print(
        #             '++++++++++++++WARNING++++++++++++++ you are barely training to test some things'
        #         )
        #         break

    return metrics_dict

def test(best_acc, args, net, device, test_loader, optimizer, epoch, freeze_classifier_with_val=False):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    stored_per_x_classifier, stored_metrics_classifier = get_empty_storage_metrics(args.G)
    stored_per_x_gate, stored_metrics_gate = get_empty_storage_metrics(args.G)
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
        if freeze_classifier_with_val:
            for idx in range(args.G):
                classifier_accuracy = compute_gated_accuracy(stored_metrics_classifier, idx)
                accuracy_tracker = net.module.accuracy_trackers[idx]
                if accuracy_tracker.should_freeze(classifier_accuracy):
                    net.module.freeze_intermediate_classifier(idx)
                    print(f"FREEZING CLASSIFIER {idx}")
                else:
                    accuracy_tracker.insert_acc(classifier_accuracy)
        if args.use_mlflow:
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
            mlflow.log_metrics(log_dict, step=batch_idx + (epoch * len(test_loader)))
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
    if args.use_mlflow:
        log_dict = {'best/test_acc': gated_acc}
        mlflow.log_metrics(log_dict)
    return stored_metrics_classifier, best_acc
