import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

# Fetch data
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from poshhoc_fmri import get_processed_input, ari_inference, get_pivotal_stats_shifted
from sanssouci.reference_families import shited_template
from sanssouci.lambda_calibration import calibrate_jer, get_pivotal_stats_shifted
from tqdm import tqdm
from scipy import stats
import sanssouci as sa

seed = 42
seed = 42
alpha = 0.05
B = 1000
k_max = 1000
k_min = 27
smoothing_fwhm = 4

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates_kmin = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=1000, seed=seed)
learned_templates_kmin[:, :k_min] = np.zeros((B, k_min))


if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']


def compute_bounds_comparison(task1s, task2s, learned_templates,
                   alpha, TDP, k_max, B,
                   smoothing_fwhm=4, n_jobs=1, seed=None, k_min=0):
    """
    Find largest FDP controlling regions on a list of contrast pairs
    using ARI, calibrated Simes and  learned templates.

    Parameters
    ----------

    task1s : list
        list of contrasts
    task2s : list
        list of contrasts
    learned_templates : array of shape (B_train, p)
        sorted quantile curves computed on training data
    alpha : float
        risk level
    k_max : int
        threshold families length
    B : int
        number of permutations at inference step
    smoothing_fwhm : float
        smoothing parameter for fMRI data (in mm)
    n_jobs : int
        number of CPUs used for computation. Default = 1

    Returns
    -------

    bounds_tot : matrix
        Size of largest FDP controlling regions for all three methods

    """
    notip_bounds = []
    pari_bounds = []

    for i in tqdm(range(len(task1s))):
        fmri_input, nifti_masker = get_processed_input(
                                                task1s[i], task2s[i],
                                                smoothing_fwhm=smoothing_fwhm)

        stats_, p_values = stats.ttest_1samp(fmri_input, 0)
        p = fmri_input.shape[1]
        _, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)
        pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                           k_max=k_max, B=B,
                                           n_jobs=n_jobs, seed=seed)
        
        shifted_templates = np.array([lambd*shifted_template(p, p, k_min=k_min) for lambd in np.linspace(0, 1, 1000)])
        calibrated_shifted_template = calibrate_jer(alpha, shifted_templates,
                                            pval0, k_min)
        calibrated_tpl = calibrate_jer(alpha, learned_templates,
                                          pval0, k_max, k_min=k_min)

        _, region_size_notip = sa.find_largest_region(p_values, calibrated_tpl,
                                                      TDP,
                                                      nifti_masker)

        _, region_size_pari = sa.find_largest_region(p_values,
                                                    calibrated_shifted_template,
                                                    TDP,
                                                    nifti_masker)

        notip_bounds.append(region_size_notip)
        pari_bounds.append(region_size_pari)

    bounds_tot = np.vstack([notip_bounds, pari_bounds])
    return bounds_tot


# Compute largest region sizes for 3 possible TDP values

res_01 = compute_bounds_comparison(test_task1s, test_task2s, learned_templates_kmin, alpha,
                        0.95, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed, k_min=k_min)

res_02 = compute_bounds_comparison(test_task1s, test_task2s, learned_templates_kmin, alpha,
                        0.9, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed, k_min=k_min)

res_03 = compute_bounds_comparison(test_task1s, test_task2s, learned_templates_kmin, alpha,
                        0.8, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed, k_min=k_min)

# multiple boxplot code adapted from
# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots


def gen_boxplot_data(res):
    idx_ok = np.where(res[0] > 25)[0]  # exclude 3 tasks with trivial sig
    power_change_notip = ((res[0] - res[1]) / res[1]) * 100
    return [power_change_notip[idx_ok]]


es_01 = [np.array([30, 40, 50]), np.array([20, 30, 40])]
res_02 = [np.array([35, 45, 55]), np.array([25, 35, 45])]
res_03 = [np.array([25, 35, 45]), np.array([15, 25, 35])]

data_a = gen_boxplot_data(res_01)
data_b = gen_boxplot_data(res_02)
data_c = gen_boxplot_data(res_03)

ticks = ['Notip with kmin vs pARI']


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


fig, ax = plt.subplots(figsize=(10, 6))

# Add dots to boxplots
for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        pos0 = np.array(range(len(data_a)))*3.0-0.4
        x = np.random.normal(pos0[nb], 0.1)
        ax.scatter(x, y, c='#66c2a4', alpha=0.75, marker='v')

for nb in range(len(data_b)):
    for i in range(len(data_b[nb])):
        y = data_b[nb][i]
        pos1 = np.array(range(len(data_b)))*3.0+0.4
        x = np.random.normal(pos1[nb], 0.1)
        ax.scatter(x, y, c='#238b45', alpha=0.75, marker='D')

for nb in range(len(data_c)):
    for i in range(len(data_c[nb])):
        y = data_c[nb][i]
        pos2 = np.array(range(len(data_c)))*3.0+1.2
        x = np.random.normal(pos2[nb], 0.1)
        ax.scatter(x, y, c='#00441b', alpha=0.75, marker='p')

bpl = ax.boxplot(data_a, positions=np.array(range(len(data_a)))*3.0-0.4,
                 sym='', widths=0.6)
bpr = ax.boxplot(data_b, positions=np.array(range(len(data_b)))*3.0+0.4,
                 sym='', widths=0.6)
bpc = ax.boxplot(data_c, positions=np.array(range(len(data_c)))*3.0+1.2,
                 sym='', widths=0.6)
set_box_color(bpl, '#66c2a4')  # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#238b45')
set_box_color(bpc, '#00441b')

# draw temporary red and blue lines and use them to create a legend
ax.scatter([], [], c='#66c2a4', marker='v', label=r'$FDP \leq 0.05$')
ax.scatter([], [], c='#238b45', marker='D', label=r'$FDP \leq 0.1$')
ax.scatter([], [], c='#00441b', marker='p', label=r'$FDP \leq 0.2$')
ax.legend(loc='upper right', prop={'size': 8.5})

ax.set_xticks(range(0, len(ticks) * 3, 3))
ax.set_xticklabels(ticks)
ax.set_ylim(-10, 80)
ax.set_ylabel('Detection rate variation (%)')
ax.hlines(0, xmin=-1.5, xmax=8, color='black')
ax.set_title(r'Detection rate variation for $\alpha = 0.05$ and various FDPs')
plt.tight_layout()
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
plt.savefig('/home/onyxia/work/Notip/figures/comparison_Notip_pARI.png')
plt.show()
