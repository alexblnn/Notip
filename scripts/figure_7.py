import numpy as np
import matplotlib.pyplot as plt
from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks
from scipy import stats
import pandas as pd

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42
B = 1000

train_task1prime = 'task001_vertical_checkerboard_vs_baseline'
train_task2prime = 'task001_horizontal_checkerboard_vs_baseline'

learned_templates = get_data_driven_template_two_tasks(train_task1prime, train_task2prime, B=B, cap_subjects=True, seed=seed)

seed = 42
alpha = 0.05
TDP = 0.9
k_max = 1000
smoothing_fwhm = 4

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

res = compute_bounds(test_task1s, test_task2s, learned_templates, alpha, TDP, k_max, B, smoothing_fwhm=smoothing_fwhm, seed=seed)

idx_ok = np.where(res[0] > 25)[0]
# reminder : this excludes 3 pathological contrast pairs with unsignificant signal

power_change_simes = ((res[1][idx_ok] - res[0][idx_ok]) / res[0][idx_ok]) * 100
power_change_learned_Simes = ((res[2][idx_ok] - res[1][idx_ok]) / res[1][idx_ok]) * 100
power_change_learned_ARI = ((res[2][idx_ok] - res[0][idx_ok]) / res[0][idx_ok]) * 100

data_a = [power_change_simes, power_change_learned_ARI, power_change_learned_Simes]
for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        x = np.random.normal(nb + 1, 0.05)
        plt.scatter(x, y, alpha=0.65, c='blue')

plt.boxplot(data_a, sym='')
plt.xticks([1, 2, 3], ['Calibrated Simes \n vs ARI', 'Learned vs ARI', 'Learned vs \n Calibrated Simes'])
plt.ylabel('Detection rate variation (%)')
plt.ylim(-15, 75)
plt.hlines(0, xmin=0.5, xmax=3.5, color='black')
plt.title(r'Detection rate variation for $\alpha = 0.05, FDP \leq 0.1$')
plt.savefig(os.path.join(fig_path, 'figure_7.pdf'))