import numpy as np
from scipy import stats
from tqdm import tqdm
import pandas as pd

from posthoc_fmri import get_processed_input
from posthoc_fmri import ari_inference
import sanssouci as sa
from joblib import Parallel, delayed
import multiprocessing
from functools import partial

import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

num_cores = multiprocessing.cpu_count()


seed = 42

alpha = 0.05
TDP = 0.95
B = 1000

df_tasks = pd.read_csv('contrast_list2.csv', index_col=0)

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
learned_templates = np.load("template10000.npy", mmap_mode="r")

pvals_perm_tot = np.load("pvals_perm_tot.npy", mmap_mode="r")

p = pvals_perm_tot.shape[2]

k_maxs = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, p]


def compute_regions(k_max, pvals_perm, p_values, alpha, TDP, nifti_masker, task_idx):
    piv_stat = sa.get_pivotal_stats(pvals_perm_tot[task_idx], K=k_max)
    lambda_quant = np.quantile(piv_stat, alpha)
    simes_thr = sa.linear_template(lambda_quant, k_max, p)

    calibrated_tpl = sa.calibrate_jer(alpha, learned_templates, pvals_perm_tot[task_idx], k_max)

    _, region_size_simes = sa.find_largest_region(p_values, simes_thr,
                                                  TDP,
                                                  nifti_masker)

    _, region_size_learned = sa.find_largest_region(p_values, calibrated_tpl,
                                                    TDP,
                                                    nifti_masker)
    return np.array([region_size_ARI, region_size_simes, region_size_learned])


for i in tqdm(range(len(test_task1s))):
    fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])
    stats_, p_values = stats.ttest_1samp(fmri_input, 0)

    _, region_size_ARI = ari_inference(p_values, TDP, alpha, nifti_masker)

    compute_regions_ = partial(compute_regions, pvals_perm=pvals_perm_tot, p_values=p_values, alpha=alpha, TDP=TDP, nifti_masker=nifti_masker, task_idx=i)
    k_max_curve = Parallel(n_jobs=num_cores)(delayed(compute_regions_)(k_max) for k_max in k_maxs)
    np.save("../figures/fig10/kmax_curve_task%d_tdp%.2f_alpha%.2f" % (i, TDP, alpha), k_max_curve)
