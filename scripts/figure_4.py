import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

# Fetch data
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import compute_bounds, get_data_driven_template_two_tasks

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
B = 1000
k_max = 1000
smoothing_fwhm = 4

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']


# Compute largest region sizes for 3 possible TDP values

res_01 = compute_bounds(test_task1s, test_task2s, learned_templates, alpha,
                        0.95, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed)

res_02 = compute_bounds(test_task1s, test_task2s, learned_templates, alpha,
                        0.9, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed)

res_03 = compute_bounds(test_task1s, test_task2s, learned_templates, alpha,
                        0.8, k_max, B, smoothing_fwhm=smoothing_fwhm,
                        n_jobs=n_jobs,
                        seed=seed)

# multiple boxplot code adapted from
# https://stackoverflow.com/questions/16592222/matplotlib-group-boxplots


def gen_boxplot_data(res):
    idx_ok = np.where(res[0] > 25)[0]  # exclude 3 tasks with trivial sig
    power_change_simes = ((res[1] - res[0]) / res[0]) * 100
    power_change_learned_Simes = ((res[2] - res[1]) / res[1]) * 100
    power_change_learned_ARI = ((res[2] - res[0]) / res[0]) * 100
    return [power_change_simes[idx_ok], power_change_learned_ARI[idx_ok],
            power_change_learned_Simes[idx_ok]]


data_a = gen_boxplot_data(res_01)
data_b = gen_boxplot_data(res_02)
data_c = gen_boxplot_data(res_03)

ticks = ['Calibrated Simes \n vs ARI', 'Notip vs ARI',
         'Notip vs \n Calibrated Simes']


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)


plt.figure()

# Add dots to boxplots
for nb in range(len(data_a)):
    for i in range(len(data_a[nb])):
        y = data_a[nb][i]
        pos0 = np.array(range(len(data_a)))*3.0-0.4
        x = np.random.normal(pos0[nb], 0.1)
        plt.scatter(x, y, c='#66c2a4', alpha=0.75, marker='v')

for nb in range(len(data_b)):
    for i in range(len(data_b[nb])):
        y = data_b[nb][i]
        pos1 = np.array(range(len(data_b)))*3.0+0.4
        x = np.random.normal(pos1[nb], 0.1)
        plt.scatter(x, y, c='#238b45', alpha=0.75, marker='D')

for nb in range(len(data_c)):
    for i in range(len(data_c[nb])):
        y = data_c[nb][i]
        pos2 = np.array(range(len(data_c)))*3.0+1.2
        x = np.random.normal(pos2[nb], 0.1)
        plt.scatter(x, y, c='#00441b', alpha=0.75, marker='p')

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*3.0-0.4,
                  sym='', widths=0.6)
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*3.0+0.4,
                  sym='', widths=0.6)
bpc = plt.boxplot(data_c, positions=np.array(range(len(data_c)))*3.0+1.2,
                  sym='', widths=0.6)
set_box_color(bpl, '#66c2a4')  # colors are from http://colorbrewer2.org/
set_box_color(bpr, '#238b45')
set_box_color(bpc, '#00441b')

# draw temporary red and blue lines and use them to create a legend
plt.scatter([], [], c='#66c2a4', marker='v', label=r'$FDP \leq 0.05$')
plt.scatter([], [], c='#238b45', marker='D', label=r'$FDP \leq 0.1$')
plt.scatter([], [], c='#00441b', marker='p', label=r'$FDP \leq 0.2$')
plt.legend()

plt.xticks(range(0, len(ticks) * 3, 3), ticks)


plt.ylim(-10, 80)
plt.ylabel('Detection rate variation (%)')
plt.hlines(0, xmin=-1.5, xmax=8, color='black')
plt.title(r'Detection rate variation for $\alpha = 0.05$ and various FDPs')
plt.legend(loc=2, prop={'size': 8.5})
plt.tight_layout()
plt.savefig('/home/onyxia/work/Notip/figures/figure_4.pdf')
plt.show()
