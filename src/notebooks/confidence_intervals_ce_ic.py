


import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_helper import collect_baseline_data, get_dict_experiment, get_our_df, get_param_plot, getting_our_data, NAME


pal = sns.color_palette()
list_colors_sns = pal.as_hex()


requested_alpha = 0.05


def get_combine_for_experiment(dataset, model):
    experiment_dict = get_dict_experiment(dataset, model)
    if experiment_dict is None:
        return None, None
    list_dicts_ours = experiment_dict[NAME.our_name.value]
    total_mudaa ,end_acc,L,end_plot_acc,line_acc,line_acc_label, lambda_val_1 = get_param_plot(dataset, model)
    for dicts in list_dicts_ours:
        
        if dataset == 'cifar100' and model == 't2t_14':
            lambda_val = dicts['lambda']
            if lambda_val == lambda_val_1:
                our_point_to_display = dicts
                print('got point 1')
        else:
            our_point_to_display = dicts
        



    our_df, cov_keys_dict, C_keys_dict = get_our_df(list_dicts_ours, requested_alpha)

    our_point_ic  = np.mean(our_point_to_display['test/total_cost'])
    our_point_acc  = np.mean(our_point_to_display['test/gated_acc'])
    boosted_df, boosted_points = collect_baseline_data(experiment_dict[NAME.boo_name.value], [our_point_ic, our_point_ic], requested_alpha, cov_keys_dict, C_keys_dict)
    boosted_df['method'] = NAME.boo_name.value


    weighted_df, weighted_points = collect_baseline_data(experiment_dict[NAME.w_name.value], [our_point_ic, our_point_ic], requested_alpha, cov_keys_dict, C_keys_dict)
    weighted_df['method'] = NAME.w_name.value


    boosted_point_ic  = np.mean(boosted_points[0]['average_IC'])
    boosted_point_acc  = np.mean(boosted_points[0]['ACC'])

    weighted_point_ic  = np.mean(weighted_points[0]['average_IC'])
    weighted_point_acc  = np.mean(weighted_points[0]['ACC'])





    filtered_boosted_df = boosted_df[boosted_df['average_ACC'].between(0, end_acc)]
    filtered_our_df = our_df[our_df['average_ACC'].between(0, end_acc)]
    filtered_weighted_df = weighted_df[weighted_df['average_ACC'].between(0, end_acc)]
    #filtered_baseline_df   = baseline_df[baseline_df['average_ACC'].between(end_plot_acc, end_acc)]
    # filtered_boosted_df = boosted_df
    # filtered_our_df = our_df
    # filtered_weighted_df = weighted_df

    filtered_boosted_df['average_IC']=filtered_boosted_df['average_IC']/total_mudaa
    filtered_weighted_df['average_IC']=filtered_weighted_df['average_IC']/total_mudaa
    filtered_our_df['average_IC']=filtered_our_df['average_IC']/total_mudaa
    boosted_point_ic=boosted_point_ic/total_mudaa
    weighted_point_ic=weighted_point_ic/total_mudaa
    our_point_ic = our_point_ic/total_mudaa

    combined_df = pd.concat([filtered_boosted_df, filtered_weighted_df, filtered_our_df],axis=0, ignore_index=True)
    return  combined_df, [[boosted_point_ic,weighted_point_ic, our_point_ic],[boosted_point_acc,weighted_point_acc, our_point_acc]]



appendix_mode = False
for dataset in ['cifar10','svhn','cifar100','cifar100LT']:
    for model in ['t2t_7', 't2t_14']:
        combined_df,  list_points=get_combine_for_experiment(dataset, model)
        
        if combined_df is not None:
            combined_df['ACC'] = combined_df['ACC']/100
            combined_df['ECE'] = combined_df['ECE']/100
            total_mudaa ,end_acc,L,end_plot_acc,line_acc,line_acc_label, lambda_val_1 = get_param_plot(dataset, model)
            plt.figure(figsize=(5,5.5))
            sns.set_theme(style="whitegrid",font_scale=1.4)
            sns.lineplot(data=combined_df, x="average_IC", y="ACC", hue="method", style="method",
                markers=True, dashes=False)
            plt.axhline(y=line_acc/100, color='k',linestyle='--', label=line_acc_label)
            
            
            x_string = r'Mul-Add ('+str(total_mudaa)+'$ \\times 10^6$)'
            plt.xlabel(x_string)
            if appendix_mode:
                plt.ylabel('Accuracy')
                plt.legend()
            else:
                if dataset== 'cifar100':
                    plt.scatter(list_points[0],[a/100 for a in list_points[1]], s=200, color='k', marker="*")
                    plt.ylabel('Accuracy')
                    
                else:
                    plt.ylabel('')
                if dataset == 'svhn':
                    plt.ylim([0.7,0.95])
                    plt.xlim([0.25,0.85])
                    #plt.legend([],[], frameon=False)
                elif dataset== 'cifar100':
                    plt.ylim([0.65,0.9])
                    plt.xlim([0.4,0.9])
                elif dataset == 'cifar100LT':
                    plt.ylim([0.5,0.95])
                    plt.xlim([0.2,0.95])
                    #plt.legend([],[], frameon=False)
                    
            plt.legend()
            plt.tight_layout()
#             plt.show()
#             plt.close()
            plt.savefig('figures/perf_cost_'+dataset+'_'+model+'.pdf')




            

            plt.figure(figsize=(5,4))
            sns.set_theme(style="whitegrid",font_scale=1.5)
            sns.lineplot(data=combined_df, x="average_IC", y="ECE", hue="method", style="method",
                markers=True, dashes=False)

            x_string = r'Mul-Add ('+str(total_mudaa)+'$ \\times 10^6$)'
            plt.xlabel(x_string)
            plt.ylabel('ECE')
            plt.tight_layout()
            
            plt.savefig('figures/perf_ece_'+dataset+'_'+model+'.pdf')
#             plt.show()
#             plt.close()






keys_gated_accs = get_all_key_with(list_dicts_ours[-1], 'gated_acc_')
keys_all_acc = get_all_key_with(list_dicts_ours[-1], 'test/acc')


weighted_df_emp_vs_req = pd.DataFrame()
boosted_df_emp_vs_req = pd.DataFrame()
our_df_emp_vs_req = pd.DataFrame()
for cov_key in cov_keys_baseline:
    print(cov_key)
    alpha = float(cov_key.split('alpha_')[-1])
    boosted_df_cov = pd.DataFrame()
    boosted_df_cov['emp_alpha'] = boosted_df[cov_key]
    boosted_df_cov['requested_alpha'] = alpha
    boosted_df_emp_vs_req = pd.concat([boosted_df_cov, boosted_df_emp_vs_req],axis=0, ignore_index=True)

    df_cov = pd.DataFrame()
    df_cov['emp_alpha'] = weighted_df[cov_key]
    df_cov['requested_alpha'] = alpha
    weighted_df_emp_vs_req = pd.concat([df_cov, weighted_df_emp_vs_req],axis=0, ignore_index=True)

    df_cov = pd.DataFrame()
    df_cov['emp_alpha'] = our_df[cov_key]
    df_cov['requested_alpha'] = alpha
    our_df_emp_vs_req = pd.concat([df_cov, our_df_emp_vs_req],axis=0, ignore_index=True)








x = [0.01,0.05]
y = [0.01,0.05]
sns.set_theme(style="whitegrid")
plt.plot(x,y, color='k',label=r'$\alpha = \hat{\alpha}$')
sns.lineplot(data=boosted_df_emp_vs_req, x="requested_alpha", y="emp_alpha", label=boo_name)
sns.lineplot(data=weighted_df_emp_vs_req, x="requested_alpha", y="emp_alpha", label=w_name)
sns.lineplot(data=our_df_emp_vs_req, x="requested_alpha", y="emp_alpha", label=our_name)

plt.xlabel(r'Theoritical (requested) $\alpha$')
plt.ylabel(r' $\hat{\alpha}$')
plt.tight_layout()
plt.savefig('figures/emp_vs_theoritical_'+dataset+'.pdf')





num_exit_key = r'$|\mathcal{D}^l|$ (gate usage)'
def aggregate_acc_baseline(w_metrics, delta):
    df_acc_cum = pd.DataFrame()
    cumul_acc = 0
    for l in range(L):
        all_acc = 'ALL_ACC_PER_GATE_'+str(l)
        gated_acc = 'GATED_ACC_PER_GATE_'+str(l)
        num_exit = 'EXIT_RATE_PER_GATE_'+str(l)
        df_acc = pd.DataFrame()
        df_acc['constant'] = 5
        df_acc['all'] = w_metrics[all_acc]
        if gated_acc in w_metrics:
           
            df_acc['gated'] = w_metrics[gated_acc]
            
            df_acc['gate'] = l+delta
            df_acc[num_exit_key] = np.mean(w_metrics[num_exit])
            
            frac = np.mean(metrics[num_exit])
            g_all = np.mean(metrics[gated_acc])
            
            ammount_acc = (frac * g_all)/100.0
            cumul_acc = cumul_acc + ammount_acc
            df_acc['cumul_acc'] = cumul_acc
            df_acc_cum = pd.concat([df_acc_cum, df_acc],axis=0, ignore_index=True)
    return df_acc_cum

def aggregate_acc_ours(our_m, delta):
    df_acc_ours = pd.DataFrame()
    cumul_acc = 0
    for l in range(L):
        
        all_acc = 'test/acc'+str(l)

        gated_acc = 'test/gated_acc_'+str(l)
        percent_exit = 'test/percent_exit'+str(l)
        df_acc = pd.DataFrame()
        df_acc['all'] = our_m[all_acc]
        df_acc['constant'] = 5
        
        if gated_acc in our_m and len(our_m[gated_acc]) == 10:

            df_acc['gated'] = our_m[gated_acc]
            df_acc['gate'] = l
            frac = np.mean(our_m[percent_exit])
            g_all = np.mean(our_m[gated_acc])
            
            s = np.mean([100*p for p in our_m[percent_exit]])
            
            df_acc[num_exit_key] = s
            df_acc_ours = pd.concat([df_acc_ours, df_acc],axis=0, ignore_index=True)
            ammount_acc = (frac * g_all)/100.0
            cumul_acc = cumul_acc + ammount_acc
            df_acc['cumul_acc'] = cumul_acc
        
    return df_acc_ours
    
def plotting_point(point,b_metrics, w_metrics, prefix):
    delta = 0.1
    df_acc_ours = aggregate_acc_ours(point, delta)
    df_acc_ours['method'] = our_name      

    df_acc_boosted = aggregate_acc_baseline(b_metrics, delta)
    df_acc_boosted['method'] = boo_name      

    
   
    df_acc_weighted = aggregate_acc_baseline(w_metrics, delta*2)
    df_acc_weighted['method'] = w_name

    df_acc =  pd.concat([df_acc_boosted,df_acc_weighted, df_acc_ours],axis=0, ignore_index=True)
    
    
    
    
    sns.set(style="whitegrid", font_scale=1.6)
   
    g = sns.relplot(
        data=df_acc,
        x="gate", y="gated", hue='method',  size=num_exit_key,
         sizes=(5, 300))
    plt.ylim([0,105])
    plt.xlim([0,14])
    g.fig.set_size_inches(7,7)
    plt.axhline(np.mean(b_metrics['ACC']), color=list_colors_sns[0], linestyle='--')
    plt.axhline(np.mean(w_metrics['ACC']), color=list_colors_sns[1],linestyle= '--')
    plt.axhline(np.mean(point['test/gated_acc']), color=list_colors_sns[2], linestyle='--')
    legend1 = plt.legend([our_name +r' accuracy ', boo_name +' accuracy', w_name +' accuracy'], loc=4)
    
    plt.gca().add_artist(legend1)
    
    g.despine(left=True, bottom=True)
    g.set(xlabel ="Gates ($l$)", ylabel = r"Accuracy on exited points ($\mathcal{D}^l$)")
    g._legend.remove()
    plt.tight_layout()
    
    plt.savefig(prefix+'_'+dataset+'gated_acc.pdf')
    
    sns.set(style="whitegrid", font_scale=1.6)
    
    g = sns.relplot(
        data=df_acc,
        x="gate", y="all", hue='method',  size=num_exit_key,
         sizes=(5, 300), legend='brief')
   
    g.despine(left=True, bottom=True)
    plt.axhline(np.mean(b_metrics['ACC']), color=list_colors_sns[0], linestyle='--')
    plt.axhline(np.mean(w_metrics['ACC']), color=list_colors_sns[1],linestyle= '--')
    plt.axhline(np.mean(point['test/gated_acc']), color=list_colors_sns[2], linestyle='--')
    g.fig.set_size_inches(7,7)
    plt.ylim([0,105])
    plt.xlim([0,14])
    
    g.set(xlabel ="Gates ($l$)", ylabel = r"Accuracy on all points ($\mathcal{D}$)")
    
    plt.tight_layout()
   
    plt.savefig(prefix+'_'+dataset+'all_acc.pdf')





point = our_point_to_display
metrics = boosted_points[0]
w_metrics = weighted_points[0]
plotting_point(point,metrics, w_metrics,prefix='figures/better' )


# In[ ]:





# In[ ]:








svhn_ours = getting_our_data('svhn' ,'t2t_7')
our_df_svhn = get_our_df(svhn_ours)
cifar10_ours = getting_our_data('cifar10' ,'t2t_7')
our_df_cif10 = get_our_df(cifar10_ours)
cifar100_ours = getting_our_data('cifar100' ,'t2t_14')
our_df_cif100 = get_our_df(cifar100_ours)





df_all_ece = pd.DataFrame()
for our_df in [our_df_cif100, our_df_cif10, our_df_svhn]:  
    df_acc_ours = pd.DataFrame()
    for l in range(30):
        key = 'test/ece'+str(l)
        if key in our_df:
            df_acc = pd.DataFrame()
            eces = our_df[key]
            df_acc['ece'] = eces
            df_acc['gate'] = l+1
            df_acc_ours = pd.concat([df_acc_ours, df_acc],axis=0, ignore_index=True)
            max_l = l+1
    print(max_l)
    df_acc_ours['gate'] = df_acc_ours['gate']/max_l
    df_all_ece = pd.concat([df_acc_ours, df_all_ece],axis=0, ignore_index=True)





g= sns.lmplot(data=df_all_ece, x="gate", y="ece")
g.fig.set_size_inches(6,4)
plt.ylabel('ECE on all points ($\mathcal{D}$)')
plt.xlabel('Gates $(\\times\\frac{l}{L})$')
plt.xlim([0,1])
plt.tight_layout()
plt.savefig('figures/average_ece.pdf')


# In[ ]:





# In[ ]:





# In[ ]:




