import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)

from posthoc_fmri import compute_bounds
from posthoc_fmri import get_data_driven_template_two_tasks

# Fetch data
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42
B = 1000

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

# Learn template using only the first 15 subjects
learned_templates = get_data_driven_template_two_tasks(
                    train_task1,
                    train_task2, B=B, cap_subjects=True, seed=seed)

seed = 42
alpha = 0.05
TDP = 0.9
k_max = 1000
smoothing_fwhm = 4

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

# Load contrast list
df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

res = compute_bounds(test_task1s,
                     test_task2s, learned_templates,
                     alpha, TDP, k_max, B,
                     smoothing_fwhm=smoothing_fwhm, n_jobs=n_jobs, seed=seed)

idx_ok = np.where(res[0] > 25)[0]
# reminder : this excludes 3 pathological contrast pairs with trivial signal

# Compute detection rate variations for the 3 possible comparisons
power_change_simes = ((res[1][idx_ok] - res[0][idx_ok]) / res[0][idx_ok]) * 100
power_change_learned_Simes = ((res[2][idx_ok] - res[1][idx_ok]) / res[1][idx_ok]) * 100
power_change_learned_ARI = ((res[2][idx_ok] - res[0][idx_ok]) / res[0][idx_ok]) * 100

data_a = [power_change_simes, power_change_learned_ARI,
          power_change_learned_Simes]

# Add dots to boxplots
for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        x = np.random.normal(nb + 1, 0.05)
        plt.scatter(x, y, alpha=0.65, c='blue')

plt.boxplot(data_a, sym='')
plt.xticks([1, 2, 3], ['Calibrated Simes \n vs ARI', 'Learned vs ARI',
                       'Learned vs \n Calibrated Simes'])
plt.ylabel('Detection rate variation (%)')
plt.ylim(-10, 80)
plt.hlines(0, xmin=0.5, xmax=3.5, color='black')
plt.title(r'Detection rate variation for $\alpha = 0.05, FDP \leq 0.1$')
plt.savefig(os.path.join(fig_path, 'figure_7.pdf'))
plt.show()
