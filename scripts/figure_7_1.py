import numpy as np
from posthoc_fmri import get_data_driven_template_two_tasks

import os

if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))

seed = 42
B = 1000

train_task1prime = 'task001_vertical_checkerboard_vs_baseline'
train_task2prime = 'task001_horizontal_checkerboard_vs_baseline'

learned_templates = get_data_driven_template_two_tasks(train_task1prime, train_task2prime, B=B, cap_subjects=True, seed=seed)

np.save("template_capped.npy", learned_templates)
