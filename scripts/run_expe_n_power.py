# %%
import os
import sys
import pandas as pd

script_path = os.path.dirname(__file__)
sys.path.insert(0, script_path)
fig_path_ = os.path.abspath(os.path.join(script_path, os.pardir))
fig_path = os.path.join(fig_path_, "figures")

from posthoc_fmri import run_all_methods_power

n_jobs = 150
seed = 41
alpha = 0.1
dim = 25
n_train = 1000  # size of external data set
sig_train = 1
sig_test = 1
max_fdp_list = [0.1] 
max_fdp_list = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
B = 1000
pi0 = 0.9
FWHM = 4

if __name__ == "__main__":
    pi0 = float(sys.argv[1])
    print(pi0)
    FWHM = int(sys.argv[2])
    print(FWHM)

repeats = 1000
n_tests = [5, 10, 20, 50, 100, 200, 500, 1000]

all_dfs = []
for n_test in n_tests:
    df = run_all_methods_power(
        dim,
        FWHM,
        pi0,
        max_fdp_list=max_fdp_list,
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
    df["n_test"] = n_test
    all_dfs.append(df)

    # intermediate backup to disk for each n_test
    df.to_csv(
        os.path.join(fig_path, f"results_pi0{pi0}_fwhm{FWHM}_ntest{n_test}_dim{dim}.csv"),
        index=False,
        float_format="%.4f"
    )

results = pd.concat(all_dfs, ignore_index=True)
results.to_csv(os.path.join(fig_path, f"results_pi0{pi0}_fwhm{FWHM}_dim{dim}.csv"), index=False)
# %%
