import matplotlib.pyplot as plt
from scipy import stats

# sys.path.append('/home/alex/Documents/repos/sanssouci.python')
from posthoc_fmri import get_processed_input, calibrate_simes

import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

seed = 5

alpha = 0.1
TDP = 0.5
B = 20

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

p = fmri_input.shape[1]
k_max = int(p/2)
stats_, p_values = stats.ttest_1samp(fmri_input, 0)

pval0, simes_thr = calibrate_simes(fmri_input, alpha, k_max=p, B=B, seed=seed)

beta1 = alpha
points1 = [beta1 * (k / p) for k in range(p)]

beta2 = simes_thr[0] * p
points2 = [beta2 * (k / p) for k in range(p)]

plt.xlabel('k', fontsize=15)
plt.ylabel('p-values', fontsize=15)
for b in range(B):
    if b == B-1:
        plt.plot(pval0[b], color='black', label='Ordered permuted p-values')
    else:
        plt.plot(pval0[b], color='black')
plt.plot(points1, color='red', label='Uncalibrated Simes', linewidth=2)
plt.plot(points2, color='orange', label='Calibrated Simes', linewidth=2)
# plt.xlim(0, 5000)
# plt.ylim(0, 0.2)
plt.legend(prop={'size': 11.5})
plt.savefig('../figures/figure_2.1.pdf')

plt.figure()
plt.xlabel('k', fontsize=15)
plt.ylabel('p-values', fontsize=15)
for b in range(B):
    if b == B-1:
        plt.plot(pval0[b], color='black', label='Ordered permuted p-values')
    else:
        plt.plot(pval0[b], color='black')
plt.plot(points1, color='red', label='Uncalibrated Simes', linewidth=2)
plt.plot(points2, color='orange', label='Calibrated Simes', linewidth=2)
plt.xlim(0, 5000)
plt.ylim(0, 0.2)
plt.legend(prop={'size': 11.5})
plt.savefig('../figures/figure_2.2.pdf')
