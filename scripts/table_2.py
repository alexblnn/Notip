import numpy as np
import os
from posthoc_fmri import get_processed_input
from posthoc_fmri import get_stat_img
from posthoc_fmri import get_clusters_table_TDP
from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)
seed = 43

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

stat_img = get_stat_img(test_task1, test_task2)

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

learned_templates = np.load(os.path.join(script_path, "template10000.npy"), mmap_mode="r")

get_clusters_table_TDP(stat_img, 3, fmri_input, learned_templates, cluster_threshold=150).to_csv(os.path.join(script_path, "cluster_results.csv"), index=False)
