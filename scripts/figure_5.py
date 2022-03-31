import matplotlib.pyplot as plt
import numpy as np
from nilearn import plotting
from scipy import stats
import sanssouci as sa

import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import get_processed_input, calibrate_simes
from posthoc_fmri import ari_inference

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

learned_templates = np.load(os.path.join(script_path, "template10000.npy"),
                            mmap_mode="r")

pval0, simes_thr = calibrate_simes(fmri_input, alpha,
                                   k_max=k_max, B=B, seed=seed)

calibrated_tpl = sa.calibrate_jer(alpha, learned_templates, pval0, k_max)

z_unmasked_simes, region_size_simes = sa.find_largest_region(p_values,
                                                             simes_thr,
                                                             TDP,
                                                             nifti_masker)

x, y, z = plotting.find_xyz_cut_coords(z_unmasked_simes)


z_unmasked_ari, region_size_ari = ari_inference(p_values, TDP,
                                                alpha, nifti_masker)

plotting.plot_stat_map(z_unmasked_ari, title='ARI: FDP controlling \
region of %s voxels' % (region_size_ari), cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_5_1.pdf'))

plotting.plot_stat_map(z_unmasked_simes, title='Calibrated Simes: FDP controlling \
region of %s voxels' % (region_size_simes), cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_5_2.pdf'))

z_unmasked_cal, region_size_cal = sa.find_largest_region(p_values,
                                                         calibrated_tpl,
                                                         TDP,
                                                         nifti_masker)

plotting.plot_stat_map(z_unmasked_cal, title='Learned template: FDP controlling \
region of %s voxels' % (region_size_cal), cut_coords=(x, y, z))

plt.savefig(os.path.join(fig_path, 'figure_5_3.pdf'))
