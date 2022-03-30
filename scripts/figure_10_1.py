import numpy as np
from tqdm import tqdm
import pandas as pd

import sys
import sanssouci as sa
import os

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, 'figures')

sys.path.append(script_path)
from posthoc_fmri import get_processed_input

fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

seed = 42

B = 1000
alpha = 0.05

df_tasks = pd.read_csv(os.path.join(script_path, 'contrast_list2.csv'))

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
fmri_input, nifti_masker = get_processed_input(test_task1s[0], test_task2s[0])
p = fmri_input.shape[1]

pvals_perm_tot = np.zeros((len(test_task1s), B, p))

for i in tqdm(range(len(test_task1s))):

    fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])
    p = fmri_input.shape[1]
    pval0 = sa.get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed)
    pvals_perm_tot[i] = pval0

np.save(os.path.join(script_path, "pvals_perm_tot.npy"), pvals_perm_tot)
