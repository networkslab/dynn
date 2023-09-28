import os
import pickle as pk
from enum import Enum
import numpy as np
import pandas as pd

class NAME(Enum):
   
    boo_name = 'Boostnet'
    w_name = 'L2W-DEN'
    our_name = 'JEI-DNN'
    base_name = 'baseline'

def getting_our_data(dataset, model):
    
    path = '/home/floregol/git/dynn/src/notebooks/'+dataset+'_'+model+'/'
    
    try:
        list_files = os.listdir(path)

        list_dicts_ours = []
        for file_name in list_files:

            if '.pk' in file_name and 'baseline' not in file_name and 'boosted' not in file_name and 'weighted' not in file_name:
                lambda_val = float(file_name.split('_')[-2])
                print(file_name)
                with open(os.path.join(path, file_name), 'rb') as file:
                    dicts = pk.load(file)
                dicts['lambda'] = lambda_val
                list_dicts_ours.append(dicts)
                
        return list_dicts_ours
    except Exception:
        print('dont have those', path)
        return None

def get_dict_experiment(dataset, model):
    long_name = model.split('_')[0] + '_vit_' + model.split('_')[1] # t2t_vit_x
    def load_file(name, file, experiment_dict):
        try:
            with open(file, 'rb') as file:
                    list_dict = pk.load(file)
                    experiment_dict[name] = list_dict
        except Exception:
            print('couldnt get', file)
            

    experiment_dict ={}
    load_file(NAME.boo_name.value, '/home/floregol/git/dynn/src/notebooks/'+dataset+'_'+model+'/'+long_name+'_boosted_'+dataset+'_results.pk', experiment_dict)
    load_file(NAME.base_name.value, '/home/floregol/git/dynn/src/notebooks/'+dataset+'_'+model+'/'+long_name+'_baseline_'+dataset+'_results.pk', experiment_dict)
    load_file(NAME.w_name.value, '/home/floregol/git/dynn/src/notebooks/'+dataset+'_'+model+'/'+long_name+'_weighted_'+dataset+'_results.pk', experiment_dict)
    
    list_dicts_ours = getting_our_data(dataset, model)
    if list_dicts_ours is None:
        return None
    experiment_dict[NAME.our_name.value] = list_dicts_ours
    return experiment_dict
    


def extract_metrics_we_want(metrics_dict, keys_we_want):
    metrics_we_want = {}
    for key, val in metrics_dict.items():
        if key in keys_we_want:
            metrics_we_want[key] = val
    return metrics_we_want

def get_all_cov_C(metrics):
    cov_keys_dict = {}
    C_keys_dict = {}
    for key in metrics.keys():
        if 'emp_alpha' in key:
            tokens = key.split('emp_alpha')
            alpha = tokens[1]
            prefix = tokens[0]
            
            cov_key = prefix+'emp_alpha'+alpha
            C_key = prefix+'C'+alpha
            if prefix in cov_keys_dict:
                cov_keys_dict[prefix].append(cov_key)
                C_keys_dict[prefix].append(C_key)
            else:
                cov_keys_dict[prefix] = [cov_key] 
                C_keys_dict[prefix] = [C_key]  
    return cov_keys_dict, C_keys_dict

def get_all_key_with(metrics, substring):
    keys_with_substring = []
    for key in metrics.keys():
        if substring in key:
            keys_with_substring.append(key)
    return keys_with_substring

def find_highest_cov(metrics_we_care_about, cov_keys, requested_alpha):
    alpha_max = 0
    for cov_key in cov_keys:
            alpha_val = float(cov_key.split('_')[-1])
            emp_alpha = metrics_we_care_about['average'+cov_key]
            if emp_alpha< requested_alpha:
                if alpha_max < alpha_val:
                    alpha_max = alpha_val
                    #print('switching fot', alpha_val)
    #print('highest alpha is ', alpha_max, 'with emp cov', )
    return alpha_max
            
def collect_cov(cov_keys_dict, C_keys_dict, metrics, requested_alpha ):
    for cov_name in cov_keys_dict.keys():
        
        cov_keys_baseline = cov_keys_dict[cov_name]
        C_keys_baseline = C_keys_dict[cov_name]
        for cov_key in cov_keys_baseline:
            metrics['average'+cov_key] = np.mean(metrics[cov_key])
            metrics['alpha'+cov_key] = float(cov_key.split('_')[-1])
        for C_key in C_keys_baseline:
            metrics['average'+C_key] = np.mean(metrics[C_key])
            
        alpha_max = find_highest_cov(metrics, cov_keys_baseline, requested_alpha)
        if alpha_max >0 :
           
            C_key_max = cov_name+'C_'+str(alpha_max)
            metrics['C_max_'+cov_name] = metrics[C_key_max]
            metrics['emp_alpha_'+cov_name] = alpha_max
            
        
def get_our_df(list_dicts_ours, requested_alpha):

    keys_ece = get_all_key_with(list_dicts_ours[-1], 'test/ece')
    keys_we_want = ['test/acc_exit','test/total_cost', 'test/gated_acc', 'test/gated_ece', 'test/gated_ece']
    cov_keys_dict, C_keys_dict = get_all_cov_C(list_dicts_ours[0])


    #type_of_conf = 'test/sets_general_'
    # cov_keys = cov_keys_dict[type_of_conf]
    # C_keys = C_keys_dict[type_of_conf]
    def flat(l):
        return [item for sublist in l for item in sublist]
    keys_we_want = keys_we_want + flat((cov_keys_dict.values())) + flat(C_keys_dict.values()) +keys_ece
    



    our_df = pd.DataFrame()
    for metrics in list_dicts_ours:

        metrics_we_care_about = extract_metrics_we_want(metrics, keys_we_want)
        metrics_we_care_about['average_IC'] = np.mean(metrics_we_care_about['test/total_cost'])
        metrics_we_care_about['average_ACC'] = np.mean(metrics_we_care_about['test/gated_acc'])
        metrics_we_care_about['ACC'] = metrics_we_care_about['test/gated_acc'] 

        metrics_we_care_about['ECE'] =metrics_we_care_about['test/gated_ece'] 
        collect_cov(cov_keys_dict, C_keys_dict, metrics_we_care_about, requested_alpha )
      

        df = pd.DataFrame(data=metrics_we_care_about)

        our_df = pd.concat([df, our_df],axis=0, ignore_index=True)

    our_df['method'] = NAME.our_name.value
    return our_df, cov_keys_dict, C_keys_dict




def replace_if_closer(points,closest_to_points, val, this_point):
    for i, point in enumerate(points):
        dif_now = np.abs(val-point)
        dif_prev = closest_to_points[i][1]
        if dif_now< dif_prev:
            closest_to_points[i] = (this_point, dif_now)
    return closest_to_points

def collect_baseline_data(list_dict_baseline, points, requested_alpha, cov_keys_dict, C_keys_dict ):
    baseline_df = pd.DataFrame()
    closest_to_points = [(None, 100) for _ in points]
    for metrics in list_dict_baseline:
        metrics['average_IC'] = np.mean(metrics['EXPECTED_FLOPS'])
        metrics['average_ACC'] = np.mean(metrics['ACC'])
        ic = np.mean(metrics['EXPECTED_FLOPS'])
        closest_to_points = replace_if_closer(points,closest_to_points, ic, metrics)
        collect_cov(cov_keys_dict, C_keys_dict, metrics, requested_alpha )

        df = pd.DataFrame(data=metrics)
        baseline_df = pd.concat([df, baseline_df],axis=0, ignore_index=True)
    return baseline_df, [tuple_points[0] for tuple_points in closest_to_points]


def get_param_plot(dataset, model):
    lambda_val_1 = None
    if dataset == 'cifar10': 
        if model == 't2t_7':
            total_mudaa = 13.4
            end_acc = 94.5
            end_plot_acc = 0.7*end_acc
            line_acc = end_acc*0.8
            line_acc_label =r'$80\%$ end acc'
            L=7
        else:
            total_mudaa = 45.7
            end_acc = 96.35
            end_plot_acc = 0.7*end_acc
            line_acc = end_acc*0.8
            line_acc_label =r'$80\%$ end acc'
            L=14
    elif dataset == 'cifar100':
        if model == 't2t_14':
            total_mudaa = 46
            end_acc = 88.4
            L=14
            lambda_val_1 = 1.5
            end_plot_acc = 0.7*end_acc
            line_acc = end_acc*0.8
            line_acc_label =r'$80\%$ end acc'
        else:
            total_mudaa = 13.6
            end_acc = 78.97
            L=14
            lambda_val_1 = 1.5
            end_plot_acc = 0.6*end_acc
            line_acc = end_acc*0.7
            line_acc_label =r'$70\%$ end acc'
            
    elif dataset == 'svhn':
        if model == 't2t_7':
            L=7
            total_mudaa = 4.2
            end_acc = 92
            end_plot_acc = 0.7*end_acc
            
            line_acc = end_acc*0.8
            line_acc_label =r'$80\%$ end acc'
            total_mudaa = 4.3
    elif dataset == 'cifar100LT':
        if model == 't2t_7':
            total_mudaa = 13.6
            end_acc = 100
            L=7
            end_plot_acc = 0
            line_acc = end_acc*0.7
            line_acc_label =r'$70\%$ end acc'
        else:
            total_mudaa = 46.2
            end_acc = 87.7
            L=14
            end_acc = 100
            end_plot_acc = 0
            line_acc = end_acc*0.7
            line_acc_label =r'$70\%$ end acc'

    return  total_mudaa ,end_acc,L,end_plot_acc,line_acc,line_acc_label,lambda_val_1