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
from posthoc_fmri import expe_sam_all_methods

n_jobs = -1
seed = 41
alpha = 0.1
dim = 15
FWHM = 4
sig_train = 0.05
sig_test = 0.05
n_train = 100
n_test = 100
fdr = 0.1
B = 1000

pi0s = [0.5, 0.6, 0.8, 0.9, 1]
nb_methods = 4

jers = np.zeros((len(pi0s), nb_methods))

for pi0 in pi0s:
    bounds = expe_sam_all_methods(
        dim,
        FWHM,
        pi0,
        sig_train=sig_train,
        sig_test=sig_test,
        fdr=fdr,
        alpha=alpha,
        n_train=n_train,
        n_test=n_test,
        repeats=1000,
        B=B,
        n_jobs=n_jobs,
        seed=seed,
    )
    jers[pi0s.index(pi0)] = bounds

np.save(os.path.join(fig_path, "jers_pi0_sam.npy"), jers)

