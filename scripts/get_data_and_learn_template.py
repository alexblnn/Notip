import numpy as np
from posthoc_fmri import get_data_driven_template_two_tasks
from nilearn.datasets import fetch_neurovault
import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

fetch_neurovault(max_images=1000000, collection_id=1952)
seed = 42

alpha = 0.1
TDP = 0.9
B = 1000

train_task1prime = 'task001_vertical_checkerboard_vs_baseline'
train_task2prime = 'task001_horizontal_checkerboard_vs_baseline'

learned_templates = get_data_driven_template_two_tasks(train_task1prime, train_task2prime, B=10000, seed=seed)

np.save("template10000.npy", learned_templates)
