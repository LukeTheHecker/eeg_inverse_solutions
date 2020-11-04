import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve
from scipy.stats import pearsonr
from util import *

def auc_eval(y_true, y_est, simSettings, pos, n_redraw = 25, plotme=False):

    betas = np.linspace(1.0, 0.0, 100)
    
    auc_close = np.zeros((n_redraw))
    auc_far = np.zeros((n_redraw))
    
    numberOfActiveSources = int(np.sum(simSettings['sourceMask']))
    numberOfDipoles = pos.shape[0]
    # Draw from the 20% of closest dipoles to sources (~100)
    closeSplit = int(round(numberOfDipoles / 5))
    # Draw from the 50% of furthest dipoles to sources
    farSplit = int(round(numberOfDipoles / 2))
    distSortedIndices = find_indices_close_to_source(simSettings, pos)
    sourceIndices = np.where(simSettings['sourceMask']==1)[0]
    
    y_true /= np.max(y_true)
    y_true[y_true>0] = 1
    if all(y_true == 1):
        y_true = simSettings['sourceMask']
    y_est /= np.max(y_est)
    
    for n in range(n_redraw):
        
        selectedIndicesClose = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[:closeSplit], size=numberOfActiveSources) ])
        selectedIndicesFar = np.concatenate([sourceIndices, np.random.choice(distSortedIndices[-farSplit:], size=numberOfActiveSources) ])
        # print(f'redraw {n}:\ny_true={y_true[selectedIndicesClose]}\y_est={y_est[selectedIndicesClose]}')
        fpr_close, tpr_close, _ = roc_curve(y_true[selectedIndicesClose], y_est[selectedIndicesClose])
   
        fpr_far, tpr_far, _  = roc_curve(y_true[selectedIndicesFar], y_est[selectedIndicesFar])
        
        auc_close[n] = auc(fpr_close, tpr_close)
        auc_far[n] = auc(fpr_far, tpr_far)
    
    auc_far = np.mean(auc_far)
    auc_close = np.mean(auc_close)
    
    if plotme:
        print("plotting")
        plt.figure()
        plt.plot(fpr_close, tpr_close, label='ROC_close')
        plt.plot(fpr_far, tpr_far, label='ROC_far')
        # plt.xlim(1, )
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'AUC_close={auc_close:.2f}, AUC_far={auc_far:.2f}')
        plt.legend()
        plt.show()
    

    return auc_close, auc_far


def eval_estimate(y_true, y_pred):
    error = np.mean(np.square(y_true-y_pred))
    error_normed = np.mean(np.square(y_true/np.max(y_true)-y_pred/np.max(y_pred)))
    corr = pearsonr(y_true, y_pred)[0]
    print(f'error={error:.3f}({error_normed:.3f}), r={corr:.2f}')
    return error, corr
