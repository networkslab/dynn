from sklearn.linear_model import LogisticRegression
from incremental_dataloader import get_cifar_10_dataloaders, get_mnist_dataloaders
import torch
import matplotlib.pyplot as plt
import numpy as np
# from src.utils import progress_bar

class LogisticReg(torch.nn.Module):
    def __init__(self,):
        super().__init__()

def get_incremental_noisy_cifar_loaders(number_of_loaders = 10, noise_step = 0.05):
    loader_tuples = []
    for i in range(number_of_loaders):
        loader_tuples.append(get_cifar_10_dataloaders(
            test_batch_size=1000,
            noise_var=noise_step * number_of_loaders - i,
            flatten=True
        )) # from noisiest to least noisy
    return loader_tuples

def get_incremental_noisy_mnist_loaders(number_of_loaders = 10, noise_step = 0.05):
    loader_tuples = []
    for i in range(number_of_loaders):
        loader_tuples.append(get_mnist_dataloaders(
            train_batch_size=1000,
            test_batch_size=1000,
            noise_var=noise_step * number_of_loaders - i,
            flatten=True
        )) # from noisiest to least noisy
    return loader_tuples

def compare_models(num_tasks, noise_step):
    MAX_ITER = 1000

    incremental_logistic_reg = LogisticRegression(n_jobs=1, warm_start=True, penalty=None, max_iter=MAX_ITER) # is not reinitialized
    partial_logistic_reg = LogisticRegression(n_jobs=1, warm_start=True, penalty=None, max_iter=MAX_ITER)
    loader_tuples = get_incremental_noisy_mnist_loaders(number_of_loaders=num_tasks, noise_step=noise_step)
    incremental_scores = []
    partial_scores = []
    fresh_scores = []
    for t in range(num_tasks):
        fresh_logistic_reg = LogisticRegression(n_jobs = 1, warm_start=False, penalty=None, max_iter=MAX_ITER) # warm start needs to be true so we can iterate over dataloader
        train_loader, test_loader = loader_tuples[t]
        for X, y in train_loader:
            fresh_logistic_reg.fit(X, y)
            incremental_logistic_reg.fit(X, y)
            partial_logistic_reg.fit(X, y)
            break

        for X, y in test_loader:
            fresh_score = fresh_logistic_reg.score(X, y)
            incremental_score = incremental_logistic_reg.score(X, y)
            partial_reset_score = partial_logistic_reg.score(X, y)
            fresh_scores.append(fresh_score)
            incremental_scores.append(incremental_score)
            partial_scores.append(partial_reset_score)
            print(f'Incremental score is {incremental_score}, fresh score {fresh_score}, partial reset score {partial_reset_score}')
            break
        reinitialize_partial_weights(partial_logistic_reg, perc=0.1)
    return incremental_scores, partial_scores, fresh_scores

def reinitialize_partial_weights(partial_log_reg, perc=0.1):
    coefs = getattr(partial_log_reg, "coef_", None)
    if coefs is not None:
        coefs_to_keep = np.random.binomial(1, 1 - perc, coefs.shape)
        new_coefs = coefs * coefs_to_keep
    setattr(partial_log_reg, 'coef_', new_coefs)

def get_scores_with_confidence_intervals(trials = 5):
    num_tasks = 30
    incremental_scores_acc = np.empty((trials, num_tasks))
    partial_scores_acc = np.empty((trials, num_tasks))
    fresh_scores_acc = np.empty((trials, num_tasks))

    for exp in range(trials):
        incremental_scores, partial_scores, fresh_scores = compare_models(num_tasks, 0.1)
        incremental_scores_acc[exp] = np.array(incremental_scores, ndmin=1)
        partial_scores_acc[exp] = np.array(partial_scores, ndmin=1)
        fresh_scores_acc[exp] = np.array(fresh_scores, ndmin=1)
    incremental_scores_mean = np.mean(incremental_scores_acc, axis=0)
    partial_scores_mean = np.mean(partial_scores_acc, axis=0)
    fresh_scores_mean = np.mean(fresh_scores_acc, axis=0)

    incremental_scores_std = np.std(incremental_scores_acc, axis = 0)
    partial_scores_std = np.std(partial_scores_acc, axis = 0)
    fresh_scores_std = np.std(fresh_scores_acc, axis = 0)
    fig, ax = plt.subplots()
    steps = np.arange(num_tasks)
    ax.errorbar(steps, incremental_scores_mean,yerr=incremental_scores_std, label='Continual')
    ax.errorbar(steps, fresh_scores_mean,yerr=fresh_scores_std, label='Reinitialized')
    ax.errorbar(steps, partial_scores_mean,yerr=partial_scores_std, label='Partial reinit')
    ax.set_xlabel('Task number')
    ax.set_ylabel('Accuracy')
    ax.set_title('Comparison of accuracies for changing tasks')
    plt.legend()
    plt.show()

get_scores_with_confidence_intervals(trials=4)

#TODO make moving task more significant
# Use seaborn
