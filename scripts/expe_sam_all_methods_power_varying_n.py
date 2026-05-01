# %%
import os
import sys
import numpy as np

from nilearn.datasets import fetch_neurovault

# for interactive use
script_path = os.getcwd()
os.chdir("scripts")

# for non interactive use
script_path = os.path.dirname(__file__)
sys.path.insert(0, script_path)

fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, "figures")

# fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

from posthoc_fmri import run_all_methods_power

n_jobs = 96
seed = 41
alpha = 0.1
dim = 15
sig_train = 0.05
sig_test = 0.05
n_train = 1000  # size of external data set
fdr = 0.1
B = 1000

pi0 = 0.9
FWHM = 4

if __name__ == "__main__":
 pi0 = float(sys.argv[1])
 print(pi0)
 FWHM = int(sys.argv[2])

repeats = 1000

n_tests = [5, 10, 20, 50, 100, 200, 500, 1000]
nb_methods = 7

repeats = 1000

jers = np.zeros((len(n_tests), nb_methods))
powers = np.zeros((len(n_tests), nb_methods))
n_features = np.zeros((len(n_tests), repeats))

for i, n_test in enumerate(n_tests):
    jer_, power_, n_features_ = run_all_methods_power(
        dim,
        FWHM,
        pi0,
        sig_train=sig_train,
        sig_test=sig_test,
        fdr=fdr,
        alpha=alpha,
        n_train=n_train,
        n_test=n_test,
        repeats=repeats,
        B=B,
        n_jobs=n_jobs,
        seed=seed,
    )
    jers[i] = jer_
    powers[i] = power_
    n_features[i] = n_features_
    np.save(
        os.path.join(fig_path, f"jers_n_sam_pi0{pi0}_fwhm{FWHM}_ntest{n_test}.npy"),
        jers,
    )
    np.save(
        os.path.join(fig_path, f"powers_n_sam_pi0{pi0}_fwhm{FWHM}_ntest{n_test}.npy"),
        powers,
    )
    np.save(
        os.path.join(
            fig_path, f"n_features_n_sam_pi0{pi0}_fwhm{FWHM}_ntest{n_test}.npy"
        ),
        n_features,
    )

np.save(os.path.join(fig_path, f"jers_n_sam_pi0{pi0}_fwhm{FWHM}.npy"), jers)
np.save(os.path.join(fig_path, f"powers_n_sam_pi0{pi0}_fwhm{FWHM}.npy"), powers)
np.save(os.path.join(fig_path, f"n_features_n_sam_pi0{pi0}_fwhm{FWHM}.npy"), n_features)

# %%
