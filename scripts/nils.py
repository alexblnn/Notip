#%%
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

seed = 42
alpha = 0.05
B = 1000
k_max = 1000
smoothing_fwhm = 4

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']

#%%
# vérifier que la calibrated_shiftes_simes fonctionne bien
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from joblib import Memory
from scipy import stats
import os

from nilearn.datasets import fetch_neurovault

seed = 42
alpha = 0.05
B = 1000
k_max = 1000
smoothing_fwhm = 4
n_jobs = 1
k_min = 27

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

# Fetch data
fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)

from posthoc_fmri import get_processed_input, calibrate_shifted_simes

seed = 42

location = './cachedir'
memory = Memory(location, mmap_mode='r', verbose=0)

train_task1 = 'task001_vertical_checkerboard_vs_baseline'
train_task2 = 'task001_horizontal_checkerboard_vs_baseline'

fmri_input, nifti_masker = get_processed_input(
                                                train_task1, train_task2,
                                                smoothing_fwhm=smoothing_fwhm)

stats_, p_values = stats.ttest_1samp(fmri_input, 0)
shifted_simes_thr = calibrate_shifted_simes(fmri_input, alpha,
                                            B=B,
                                            n_jobs=n_jobs, seed=seed)
#%%
import matplotlib.pyplot as plt
from posthoc_fmri import calibrate_shifted_simes
plt.plot(shifted_simes_thr)
plt.show()
