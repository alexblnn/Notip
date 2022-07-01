import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy import stats
import sanssouci as sa
from joblib import Memory

import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import compute_bounds_single_task


seed = 42
B = 1000

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

optimal_task1 = 'task001_vertical_checkerboard_vs_baseline'
optimal_task2 = 'task001_horizontal_checkerboard_vs_baseline'

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

res = compute_bounds_single_task(test_task1s,
                     test_task2s,
                     alpha, TDP, k_max, B=B,
                     smoothing_fwhm=smoothing_fwhm, n_jobs=n_jobs, seed=seed)

res_opti = compute_bounds_single_task([optimal_task1],
                     [optimal_task2],
                     alpha, TDP, k_max, B=B,
                     smoothing_fwhm=smoothing_fwhm, n_jobs=n_jobs, seed=seed)


# Compute detection rate variations for the 3 possible comparisons
power_change_learned_Simes = ((res[2] - res[1]) / res[1]) * 100
power_change_learned_ARI = ((res[2] - res[0]) / res[0]) * 100

data_a = [power_change_learned_ARI,
          power_change_learned_Simes]

# Add dots to boxplots
for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        x = np.random.normal(nb + 1, 0.05)
        plt.scatter(x, y, alpha=0.65, c='blue')

notip_vs_ari_opti = ((res_opti[2][0] - res_opti[0][0]) / res_opti[0][0]) * 100
notip_vs_simes_opti = ((res_opti[2][0] - res_opti[1][0]) / res_opti[1][0]) * 100
plt.scatter(1, notip_vs_ari_opti, c='green', label='Optimal template')
plt.scatter(2, notip_vs_simes_opti, c='green')

plt.boxplot(data_a, sym='')
plt.xticks([1, 2], ['Notip vs ARI',
                       'Notip vs \n Calibrated Simes'])
plt.ylabel('Detections variation (%)')
# plt.ylim(-10, 80)
plt.hlines(0, xmin=0.5, xmax=2.5, color='black')
plt.title(r'Variation of the number of detections for $\alpha = 0.05, FDP \leq 0.1$')
plt.legend()
plt.savefig(os.path.join(fig_path, 'figure_10.pdf'))
plt.show()
