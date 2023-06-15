
import matplotlib.pyplot as plt

def generate_thresholding_plots(threhsold_name, all_sorted_p_max, all_cumul_acc, all_correct, min_x, target_acc, thresholds):
    G = len(all_sorted_p_max)
    if G>0:
        fig, axs = plt.subplots(G)
        fig.suptitle('Thresholds from sorted p max')


        for g in range(len(all_sorted_p_max)):
            sorted_p_max = all_sorted_p_max[g] 
            cumul_acc = all_cumul_acc[g] 
            correct = all_correct[g] 
            axs[g].plot(sorted_p_max, cumul_acc)
            axs[g].vlines(sorted_p_max[min_x], 0, max(cumul_acc), 'k')
        
            if thresholds[g] < 1:
                axs[g].vlines(thresholds[g], 0, max(cumul_acc), 'r')
        axs[g].set(ylabel='cumacc  {}'.format(g), xlabel='p max')
        #plt.tight_layout()
        plt.savefig('threshold'+threhsold_name+'.pdf')
        plt.close()
    