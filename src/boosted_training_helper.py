'''Train BOOSTED baseline from checkpoint of trained backbone'''

import torch
import mlflow
from timm.models import *
from timm.models import create_model
from learning_helper import get_boosted_loss
from utils import fix_the_seed, progress_bar

def train_boosted(args, net, device, train_loader, optimizer, epoch):
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        boosted_loss = get_boosted_loss(inputs, targets, optimizer, net.module)
        boosted_loss.backward()
        optimizer.step()
        progress_bar(
            batch_idx, len(train_loader),
            'Classifier Loss: %.3f' % boosted_loss)


def test_boosted(args, net, test_loader, epoch):
    net.eval()
    n_blocks = len(net.module.blocks)
    corrects = [0] * n_blocks
    totals = [0] * n_blocks
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()
        with torch.no_grad():
            preds = net.module.forward(x, n_blocks-1)
   
        for i, pred in enumerate(preds):
            corrects[i] += (torch.argmax(pred, 1) == y).sum().item()
            totals[i] += y.shape[0]
    corrects = [c / t * 100 for c, t in zip(corrects, totals)]
    log_dict = {}
    for blk in range(n_blocks):
        log_dict['test' + '/accuracies' +
                 str(blk)] = corrects[blk]
    mlflow.log_metrics(log_dict)
    return corrects

