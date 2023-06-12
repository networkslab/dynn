# Training

from utils import progress_bar
import torch
import mlflow

def collect_metrics(name,
                    epoch,
                    loader,
                    net,
                    optimizer,
                    criterion,
                    device,
                    use_mlflow=True):
    epoch_loss = 0
    correct = 0
    total = 0
    cheating_correct = 0
    list_cheating_correct_inter = [0 for _ in range(net.num_gates)]
    list_correct_inter = [0 for _ in range(net.num_gates)]
    store_all_preds= []
    store_all_pmax = []
    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, intermediate_outputs = net(inputs)
        loss = criterion(outputs,
                         targets)  # the grad_fn of this loss should be None
        for intermediate_output in intermediate_outputs:
            intermediate_loss = criterion(intermediate_output, targets)
            loss += intermediate_loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        correctly_classified = torch.full(predicted.eq(targets).shape,
                                          False).to(device)
        for i, _ in enumerate(list_correct_inter):
            _, predicted_inter = intermediate_outputs[i].max(1)
            correctly_classified += predicted_inter.eq(
                targets)  # getting all the corrects we can
            list_cheating_correct_inter[i] += correctly_classified.sum().item()
            list_correct_inter[i] += predicted_inter.eq(targets).sum().item()

        correctly_classified += predicted.eq(
            targets)  # getting all the corrects we can
        cheating_correct += correctly_classified.sum().item()

        progress_bar(
            batch_idx, len(loader),
            'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cheating: %.3f%%' %
            (epoch_loss / (batch_idx + 1), 100. * correct / total, correct,
             total, 100. * cheating_correct / total))

        if use_mlflow:
            log_dict = {
                name + '/loss': epoch_loss / (batch_idx + 1),
                name + '/acc': 100. * correct / total,
                name + '/cheating_acc': 100. * cheating_correct / total
            }
            for i, _ in enumerate(list_correct_inter):
                log_dict[name + '/acc' +
                         str(i)] = 100. * list_correct_inter[i] / total
                log_dict[
                    name + '/cheating_acc' +
                    str(i)] = 100. * list_cheating_correct_inter[i] / total
            mlflow.log_metrics(log_dict, step=batch_idx + (epoch * len(loader)))
        return {'acc':100. * correct / total}