import numpy as np
from scipy import stats
from tqdm import tqdm
import pandas as pd

import sanssouci as sa
from joblib import Parallel, delayed, Memory
import multiprocessing
from functools import partial

import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input
from posthoc_fmri import ari_inference, get_data_driven_template_two_tasks

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

num_cores = multiprocessing.cpu_count()

seed = 42

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=10000, seed=seed)


seed = 42

alpha = 0.05
TDP = 0.95
B = 1000

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

pvals_perm_tot = np.load(os.path.join(script_path, "pvals_perm_tot.npy"),
                         mmap_mode="r")

p = pvals_perm_tot.shape[2]

k_maxs = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, p]


def compute_regions(
        k_max, pvals_perm, p_values, alpha, TDP, nifti_masker, task_idx):
    piv_stat = sa.get_pivotal_stats(pvals_perm_tot[task_idx], K=k_max)
    lambda_quant = np.quantile(piv_stat, alpha)
    simes_thr = sa.linear_template(lambda_quant, k_max, p)

    calibrated_tpl = sa.calibrate_jer(alpha, learned_templates,
                                      pvals_perm_tot[task_idx], k_max)

    _, region_size_simes = sa.find_largest_region(p_values, simes_thr,
                                                  TDP,
                                                  nifti_masker)

    _, region_size_learned = sa.find_largest_region(p_values, calibrated_tpl,
                                                    TDP,
                                                    nifti_masker)
    return np.array([region_size_ARI, region_size_simes, region_size_learned])


for i in tqdm(range(len(test_task1s))):
    fmri_input, nifti_masker = get_processed_input(test_task1s[i],
                                                   test_task2s[i])
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)

    _, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)

    compute_regions_ = partial(compute_regions, pvals_perm=pvals_perm_tot,
                               p_values=p_values, alpha=alpha, TDP=TDP,
                               nifti_masker=nifti_masker, task_idx=i)
    k_max_curve = Parallel(n_jobs=num_cores)(
                    delayed(compute_regions_)(k_max) for k_max in k_maxs)
    np.save(os.path.join(fig_path,
            "fig10/kmax_curve_task%d_tdp%.2f_alpha%.2f" % (i, TDP, alpha)),
            k_max_curve)
