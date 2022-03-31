import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import sys

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)

from posthoc_fmri import get_processed_input, compute_bounds

# Fetch data
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42
alpha = 0.05
TDP = 0.9
B = 1000
k_max = 1000
smoothing_fwhm = 4

# Get contrast list
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list.csv'),
                       index_col=0)

# Load learned template
learned_templates = np.load(os.path.join(script_path, "template10000.npy"),
                            mmap_mode="r")

test_task1s = list(pd.concat([df_tasks['task1'], df_tasks['task3']]))
test_task2s = list(pd.concat([df_tasks['task2'], df_tasks['task4']]))
# test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

# Check number of subjects for each contrast pair
subj = []

for i in tqdm(range(len(test_task1s))):

    fmri_input, nifti_masker = get_processed_input(test_task1s[i],
                                                   test_task2s[i])

    subj.append(fmri_input.shape[0])

subj = np.array(subj)

res = compute_bounds(test_task1s, test_task2s,
                     learned_templates, alpha, TDP, k_max,
                     B, smoothing_fwhm=smoothing_fwhm, seed=seed)

diff = res[2] - res[1]

idx_pos = np.where(diff > 2)[0]
idx_neg = np.where(diff < - 2)[0]

# extract indices with a significant difference between learned and Simes
# plot the power change wrt the number of subjects

plt.scatter(subj[idx_neg], (diff[diff < -2] / res[1][idx_neg]) * 100,
            color='red')
plt.scatter(subj[idx_pos], (diff[diff > 2] / res[1][idx_pos]) * 100,
            color='green')
plt.hlines(0, xmin=0, xmax=160, color='black')
plt.xlabel('Sample size')
plt.ylabel('Detection rate variation (%)')
plt.title(r'Detection rate variation for $\alpha = 0.05, FDP \leq 0.1$')
plt.ylim(-40, 40)
plt.savefig(os.path.join(fig_path, 'figure_8.pdf'))
