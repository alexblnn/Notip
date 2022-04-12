import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy import stats
from scipy.stats import norm
import sys
from joblib import Memory

import sanssouci as sa

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input, calibrate_simes
from posthoc_fmri import bh_inference, get_data_driven_template_two_tasks
from posthoc_fmri import _compute_hommel_value

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=10000, seed=seed)

seed = 43

alpha = 0.05
TDP = 0.9
B = 1000
k_max = 1000

if len(sys.argv) > 1:
    n_jobs = int(sys.argv[1])
else:
    n_jobs = 1

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

p = fmri_input.shape[1]
stats_, p_values = stats.ttest_1samp(fmri_input, 0)

pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                   k_max=k_max, B=B, n_jobs=n_jobs, seed=seed)

z_unmasked_simes, region_size_simes = sa.find_largest_region(p_values,
                                                             simes_thr,
                                                             TDP,
                                                             nifti_masker)

x, y, z = plotting.find_xyz_cut_coords(z_unmasked_simes)

calibrated_tpl = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

z_unmasked_cal, region_size_cal = sa.find_largest_region(p_values,
                                                         calibrated_tpl,
                                                         TDP,
                                                         nifti_masker)

plotting.plot_stat_map(z_unmasked_cal, title='Learned template: FDP < 0.1',
                       cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_6_1.pdf'))
plt.show()

z_bh, region_size_bh = bh_inference(p_values, 1-TDP, nifti_masker)
plotting.plot_stat_map(z_bh, title='BH: FDR < 0.1', cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_6_2.pdf'))
plt.show()
z_vals = norm.isf(p_values)
hommel = _compute_hommel_value(z_vals, alpha)
ari_thr = sa.linear_template(alpha, hommel, hommel)

# Table 1 values
sa.min_tdp(np.sort(p_values)[:region_size_bh], ari_thr)
sa.min_tdp(np.sort(p_values)[:region_size_bh], simes_thr)
sa.min_tdp(np.sort(p_values)[:region_size_bh], calibrated_tpl)
