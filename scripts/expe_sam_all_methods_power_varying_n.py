# %%
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

from nilearn.datasets import fetch_neurovault

script_path = os.path.dirname(__file__)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, "figures")

# fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

sys.path.append(script_path)
from posthoc_fmri import expe_sam_all_methods_power

n_jobs = -1
seed = 41
alpha = 0.1
dim = 15
FWHM = 4
sig_train = 0.05
sig_test = 0.05
n_train = 100
fdr = 0.1
B = 1000
pi0 = 0.9

n_tests = [5, 10, 20, 50, 100, 200]
nb_methods = 5

jers = np.zeros((len(n_tests), nb_methods))
powers = np.zeros((len(n_tests), nb_methods))

for n_test in n_tests:
    jer_, power_ = expe_sam_all_methods_power(
        dim,
        FWHM,
        pi0,
        sig_train=sig_train,
        sig_test=sig_test,
        fdr=fdr,
        alpha=alpha,
        n_train=n_train,
        n_test=n_test,
        repeats=100,
        B=B,
        n_jobs=n_jobs,
        seed=seed,
    )
    jers[n_tests.index(n_test)] = jer_
    powers[n_tests.index(n_test)] = power_

np.save(os.path.join(fig_path, "jers_n_sam.npy"), jers)
np.save(os.path.join(fig_path, "powers_n_sam.npy"), powers)

