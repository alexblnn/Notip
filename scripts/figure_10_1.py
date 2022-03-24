import numpy as np
from tqdm import tqdm
import pandas as pd

from posthoc_fmri import get_processed_input
import sanssouci as sa

seed = 42

B = 1000
alpha = 0.05

df_tasks = pd.read_csv('contrast_list2.csv', index_col=0)

test_task1s, test_task2s = df_tasks['task1'], df_tasks['task2']
fmri_input, nifti_masker = get_processed_input(test_task1s[0], test_task2s[0])
p = fmri_input.shape[1]

pvals_perm_tot = np.zeros((len(test_task1s), B, p))

for i in tqdm(range(len(test_task1s))):

    fmri_input, nifti_masker = get_processed_input(test_task1s[i], test_task2s[i])
    p = fmri_input.shape[1]
    pval0 = sa.get_permuted_p_values_one_sample(fmri_input, B=B, seed=seed)
    pvals_perm_tot[i] = pval0

np.save("pvals_perm_tot.npy", pvals_perm_tot)
