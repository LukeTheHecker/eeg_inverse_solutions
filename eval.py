import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc

def auc_eval(y_true, y_est, plotme=False):
    
    betas = np.arange(0.01, 1, 0.01)
    sensitivity = np.zeros((len(betas)))
    specificity = np.zeros((len(betas)))

    for i, beta in enumerate(betas):
        # print(f'beta={beta}')

        y_true_tmp = y_true / np.max(y_true)
        y_est_tmp = y_est / np.max(y_est)
        
        y_true_tmp[y_true_tmp<beta] = 0
        y_est_tmp[y_est_tmp<beta] = 0

        y_true_tmp[y_true_tmp>0] = 1
        y_est_tmp[y_est_tmp>0] = 1
        
        nTruePositives = len(np.where(np.logical_and(y_true_tmp == 1, y_est_tmp == 1))[0])
        nTrueNegatives = len(np.where(np.logical_and(y_true_tmp == 0, y_est_tmp == 0))[0])
        nFalsePositives = len(np.where(np.logical_and(y_est_tmp == 1, y_true_tmp == 0))[0])
        nFalseNegatives = len(np.where(np.logical_and(y_true_tmp == 1, y_est_tmp == 0))[0])
        
        # print(f'nTruePositives={nTruePositives}')

        sensitivity[i] = nTruePositives / (nTruePositives + nFalseNegatives)
        specificity[i] = nTrueNegatives / (nTrueNegatives + nFalsePositives)
    AUC = auc(sensitivity, specificity)
    if plotme:
        print("plotting")
        plt.figure()
        plt.plot(1-specificity, sensitivity)
        # plt.xlim(1, )
        plt.xlabel('1-specificity')
        plt.ylabel('sensitivity')
        plt.title(f'AUC={AUC:.2f}')
        plt.show()
    

    return AUC

def eval_estimate(y_true, y_est):
    error = np.mean(np.square(y_true-y_est))
    error_normed = np.mean(np.square(y_true/np.max(y_true)-y_est/np.max(y_est)))
    corr = pearsonr(y_true, y_est)[0]
    print(f'error={error:.3f}({error_normed:.3f}), r={corr:.2f}')
    return error, corr