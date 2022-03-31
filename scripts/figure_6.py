import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy import stats
from scipy.stats import norm
import sys

import sanssouci as sa

import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input, calibrate_simes
from posthoc_fmri import bh_inference
from posthoc_fmri import _compute_hommel_value

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 43

alpha = 0.05
TDP = 0.9
B = 1000
k_max = 1000

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

p = fmri_input.shape[1]
stats_, p_values = stats.ttest_1samp(fmri_input, 0)

pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                   k_max=k_max, B=B, seed=seed)

z_unmasked_simes, region_size_simes = sa.find_largest_region(p_values,
                                                             simes_thr,
                                                             TDP,
                                                             nifti_masker)

x, y, z = plotting.find_xyz_cut_coords(z_unmasked_simes)

learned_templates = np.load(os.path.join(script_path, "template10000.npy"),
                            mmap_mode="r")

calibrated_tpl = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

z_unmasked_cal, region_size_cal = sa.find_largest_region(p_values,
                                                         calibrated_tpl,
                                                         TDP,
                                                         nifti_masker)

plotting.plot_stat_map(z_unmasked_cal, title='Learned template: FDP < 0.1',
                       cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_6_1.pdf'))

z_bh, region_size_bh = bh_inference(p_values, 1-TDP, nifti_masker)
plotting.plot_stat_map(z_bh, title='BH: FDR < 0.1', cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_6_2.pdf'))
z_vals = norm.isf(p_values)
hommel = _compute_hommel_value(z_vals, alpha)
ari_thr = sa.linear_template(alpha, hommel, hommel)

# Table 1 values
sa.min_tdp(np.sort(p_values)[:region_size_bh], ari_thr)
sa.min_tdp(np.sort(p_values)[:region_size_bh], simes_thr)
sa.min_tdp(np.sort(p_values)[:region_size_bh], calibrated_tpl)
