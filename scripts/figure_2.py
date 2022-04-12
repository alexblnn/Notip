import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import os
import sys
from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input, calibrate_simes

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 5

alpha = 0.1
TDP = 0.5
B = 20

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

p = fmri_input.shape[1]
stats_, p_values = stats.ttest_1samp(fmri_input, 0)

pval0, simes_thr = calibrate_simes(fmri_input, alpha, k_max=p,
                                   B=B, n_jobs=n_jobs, seed=seed)

beta1 = alpha
points1 = [beta1 * (k / p) for k in range(p)]

beta2 = simes_thr[0] * p
points2 = [beta2 * (k / p) for k in range(p)]

plt.xlabel('k', fontsize=15)
plt.ylabel('p-values', fontsize=15)
for b in range(B):
    if b == B-1:
        plt.loglog(pval0[b], color='black',
                   label='Ordered permuted p-values', linewidth=0.9)
    else:
        plt.loglog(pval0[b], color='black', linewidth=0.9)
plt.loglog(points1, color='red', label='Uncalibrated Simes', linewidth=2)
plt.loglog(points2, color='orange', label='Calibrated Simes', linewidth=2)
plt.legend(prop={'size': 11.5})
plt.savefig(os.path.join(fig_path, 'figure_2.pdf'))
plt.show()
