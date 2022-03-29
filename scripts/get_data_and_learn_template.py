import numpy as np
from posthoc_fmri import get_data_driven_template_two_tasks
from nilearn.datasets import fetch_neurovault
import os

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42

train_task1prime = 'task001_vertical_checkerboard_vs_baseline'
train_task2prime = 'task001_horizontal_checkerboard_vs_baseline'

learned_templates = get_data_driven_template_two_tasks(train_task1prime, train_task2prime, B=10000, seed=seed)

np.save(os.path.join(script_path, "template10000.npy"), learned_templates)
