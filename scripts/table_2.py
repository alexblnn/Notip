import numpy as np
from joblib import Memory

import os
import sys

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import get_processed_input, calibrate_simes, get_stat_img
from posthoc_fmri import ari_inference, get_data_driven_template_two_tasks
from posthoc_fmri import get_clusters_table_TDP

seed = 42

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

get_data_driven_template_two_tasks = memory.cache(
                                    get_data_driven_template_two_tasks)

learned_templates = get_data_driven_template_two_tasks(
                    train_task1, train_task2, B=10000, seed=seed)

test_task1 = 'task001_look_negative_cue_vs_baseline'
test_task2 = 'task001_look_negative_rating_vs_baseline'

stat_img = get_stat_img(test_task1, test_task2)

fmri_input, nifti_masker = get_processed_input(test_task1, test_task2)

get_clusters_table_TDP(stat_img, 3, fmri_input, learned_templates, cluster_threshold=150).to_csv(os.path.join(script_path, "cluster_results.csv"), index=False)
