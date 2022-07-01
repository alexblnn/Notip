import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import sim_experiment_notip

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

seed = 41
alpha = 0.05
dim = 25
FWHM = 4
pi0 = 0.9
sig_train = 0.05
sig_test = 0.05
n_train = 100
n_test = 50
fdr = 0.1
train_on_same = True
B = 1000

bounds = sim_experiment_notip(dim, FWHM, pi0,
                           sig_train=sig_train,
                           sig_test=sig_test,
                           fdr=fdr, alpha=alpha,
                           n_train=n_train,
                           n_test=n_test,
                           train_on_same=train_on_same,
                           repeats=1000, B=B,
                           n_jobs=n_jobs,
                           seed=seed)


bounds_fdp = bounds[:3]
bounds_tdp = bounds[3:]

def plot_results(bounds, alpha, fdr, n_train, n_test, FWHM, TDP=False, train_on_same=False):
    for nb in range(len(bounds)):
        for i in range(len(bounds[nb])):
            y = bounds[nb][i]
            x = np.random.normal(nb + 1, 0.05)
            plt.scatter(x, y, alpha=0.65, c='blue')

    plt.boxplot(bounds, sym='')
    if TDP:
        if train_on_same:
            plt.xticks([1, 2, 3], ['Calibrated Simes \n vs ARI', 'Notip (single dataset) \n vs ARI', 'Notip (single dataset) \n vs Calibrated Simes'])
        else:
            plt.xticks([1, 2, 3], ['Calibrated Simes \n vs ARI', 'Notip \n vs ARI', 'Notip \n vs Calibrated Simes'])
        plt.title(f'Empirical TPR for requested FDP control q = {fdr} at level α={alpha}')
        plt.ylabel('TPR variation (%)')
        plt.savefig(os.path.join(fig_path, 'figure_12.pdf'))
        

    else:
        plt.hlines(fdr, xmin=0.8, xmax=3.3, label='Requested FDP control', color='red')
        if train_on_same:
            plt.xticks([1, 2, 3], ['ARI', 'Calibrated \n Simes', 'Notip (single dataset)'])
        else:
            plt.xticks([1, 2, 3], ['ARI', 'Calibrated \n Simes', 'Notip'])
        plt.title(f'Empirical FDP for requested FDP control q = {fdr} at level α={alpha}')
        plt.ylabel('Empirical FDP')
        plt.legend(loc='best')
        plt.savefig(os.path.join(fig_path, 'figure_13.pdf'))
    
    plt.show()

plot_results(bounds_fdp, alpha, fdr, n_train, n_test, FWHM, TDP=False, train_on_same=train_on_same)
plot_results(bounds_tdp, alpha, fdr, n_train, n_test, FWHM, TDP=True, train_on_same=train_on_same)