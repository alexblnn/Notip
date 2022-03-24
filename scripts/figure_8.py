import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from posthoc_fmri import get_processed_input, compute_bounds, calibrate_simes, ari_inference, get_studies_list
from posthoc_fmri import draw_tasks_random, build_csv_study
import sanssouci as sa
import os
from scipy import stats

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

seed = 42
alpha = 0.05
TDP = 0.9
B = 1000
k_max = 1000
smoothing_fwhm = 4
df_tasks = pd.read_csv('contrast_list.csv', index_col=0)
df_tasks = pd.read_csv('contrast_list_new.csv')

learned_templates = np.load("template10000.npy", mmap_mode="r")

test_task1s = list(pd.concat([df_tasks['task1'], df_tasks['task3']]))
test_task2s = list(pd.concat([df_tasks['task2'], df_tasks['task4']]))
# test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
subj = []

for i in tqdm(range(len(test_task1s))):

    fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])

    subj.append(fmri_input.shape[0])

subj = np.array(subj)

subj
res = compute_bounds(test_task1s, test_task2s, learned_templates, alpha, TDP, k_max, B, smoothing_fwhm=smoothing_fwhm, seed=seed)
np.save("subjects_new_kmax1000.npy", res)
res = np.load("subjects_kmax1000.npy")
diff = res[2] - res[1]

idx_pos = np.where(diff > 2)[0]
idx_neg = np.where(diff < - 2)[0]

# extract indices with a significant difference between learned and Simes
# plot the power change wrt the number of subjects

plt.scatter(subj[idx_neg], (diff[diff < -2] / res[1][idx_neg]) * 100, color='red')
plt.scatter(subj[idx_pos], (diff[diff > 2] / res[1][idx_pos]) * 100, color='green')
plt.hlines(0, xmin=0, xmax=160, color='black')
plt.xlabel('Sample size')
plt.ylabel('Detection rate variation (%)')
plt.title(r'Detection rate variation for $\alpha = 0.05, FDP \leq 0.1$')
plt.ylim(-40, 40)
plt.savefig('../figures/figure_8.pdf')
