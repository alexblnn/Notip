import numpy as np
import matplotlib.pyplot as plt
from posthoc_fmri import compute_bounds, get_processed_input
import pandas as pd

import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

seed = 42
alpha = 0.05
TDP = 0.9
B = 1000
k_max = 1000
smoothing_fwhm_inference = 8

df_tasks = pd.read_csv('contrast_list2.csv', index_col=0)

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

learned_templates = np.load("template10000.npy", mmap_mode="r")
res = compute_bounds(test_task1s, test_task2s, learned_templates, alpha, TDP, k_max, B, smoothing_fwhm=smoothing_fwhm_inference, seed=seed)
np.save("res_smoothing_kmax1000.npy", res)
# res = np.load("res_smoothing_kmax1000.npy")
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
plt.ylabel('Detection rate variation')
plt.ylim(-30, 75)
plt.hlines(0, xmin=0.5, xmax=3.5, color='black')
plt.title(r'Detection rate variation for $\alpha = 0.05, FDP \leq 0.1$')
plt.savefig("../figures/figure_9.pdf")
