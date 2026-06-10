# %%
import os
import sys
import numpy as np

#from nilearn.datasets import fetch_neurovault

# for interactive use
script_path = os.getcwd()
os.chdir("scripts")

# for non interactive use
script_path = os.path.dirname(__file__)
sys.path.insert(0, script_path)

fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, "figures")

# fetch_neurovault(max_images=np.infty, mode='download_new', collection_id=1952)

#from posthoc_fmri_light import run_all_methods_power # caution, also need Field
from posthoc_fmri import run_all_methods_power

n_jobs = 150
seed = 41
alpha = 0.1
dim = 25
n_train = 1000  # size of external data set
sig_train = 1
sig_test = 1
max_fdp = 0.1
B = 1000

pi0 = 0.9
FWHM = 4

if __name__ == "__main__":
 pi0 = float(sys.argv[1])
 print(pi0)
 FWHM = int(sys.argv[2])
 print(FWHM)
repeats = 1000

n_tests = [5, 10, 20, 50, 100, 200, 500, 1000, 2000]
n_tests = [5, 10, 20, 50, 100, 200, 500, 1000]
nb_methods = 7

jers = np.zeros((len(n_tests), nb_methods))
powers = np.zeros((len(n_tests), nb_methods))
n_features = np.zeros((len(n_tests), repeats))

for i, n_test in enumerate(n_tests):
    jer_, power_, n_features_ = run_all_methods_power(
        dim,
        FWHM,
        pi0,
        max_fdp=max_fdp,
        alpha=alpha,
        n_train=n_train,
        n_test=n_test,
        sig_train=sig_train,
        sig_test=sig_test,
        repeats=repeats,
        B=B,
        n_jobs=n_jobs,
        seed=seed,
    )
    jers[i] = jer_
    powers[i] = power_
    n_features[i] = n_features_
    np.save(
        os.path.join(fig_path, f"jers_n_pi0{pi0}_fwhm{FWHM}_ntest{n_test}_dim{dim}.npy"),
        jers,
    )
    np.save(
        os.path.join(fig_path, f"powers_n_pi0{pi0}_fwhm{FWHM}_ntest{n_test}_dim{dim}.npy"),
        powers,
    )
    np.save(
        os.path.join(
            fig_path, f"n_features_n_pi0{pi0}_fwhm{FWHM}_ntest{n_test}_dim{dim}.npy"
        ),
        n_features,
    )

np.save(os.path.join(fig_path, f"jers_n_pi0{pi0}_fwhm{FWHM}_dim{dim}.npy"), jers)
np.save(os.path.join(fig_path, f"powers_n_pi0{pi0}_fwhm{FWHM}_dim{dim}.npy"), powers)
np.save(os.path.join(fig_path, f"n_features_n_pi0{pi0}_fwhm{FWHM}_dim{dim}.npy"), n_features)

# %%
